#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/*
 * Native helpers for gait event detection.
 *
 * Focus: fast interval-wise assignment for alternating 2-state paths.
 */

typedef struct {
    Py_buffer view;
    const double *ptr;
    Py_ssize_t n;
} DoubleBuf;

typedef struct {
    Py_buffer view;
    const int32_t *ptr;
    Py_ssize_t n;
} Int32Buf;

static void release_double_buf(DoubleBuf *b) {
    if (b->view.obj) {
        PyBuffer_Release(&b->view);
        b->view.obj = NULL;
    }
}

static void release_int32_buf(Int32Buf *b) {
    if (b->view.obj) {
        PyBuffer_Release(&b->view);
        b->view.obj = NULL;
    }
}

static int parse_double_buffer(PyObject *obj, DoubleBuf *out) {
    out->view.obj = NULL;
    out->ptr = NULL;
    out->n = 0;

    if (PyObject_GetBuffer(obj, &out->view, PyBUF_CONTIG_RO) != 0) {
        PyErr_SetString(PyExc_TypeError, "expected a contiguous numeric buffer (double)");
        return 0;
    }
    if (out->view.itemsize != (Py_ssize_t)sizeof(double) || out->view.len % (Py_ssize_t)sizeof(double) != 0) {
        PyErr_SetString(PyExc_TypeError, "buffer must contain float64 values");
        release_double_buf(out);
        return 0;
    }
    out->ptr = (const double *)out->view.buf;
    out->n = out->view.len / (Py_ssize_t)sizeof(double);
    return 1;
}

static int parse_int32_buffer(PyObject *obj, Int32Buf *out) {
    out->view.obj = NULL;
    out->ptr = NULL;
    out->n = 0;

    if (PyObject_GetBuffer(obj, &out->view, PyBUF_CONTIG_RO) != 0) {
        PyErr_SetString(PyExc_TypeError, "expected a contiguous numeric buffer (int32)");
        return 0;
    }
    if (out->view.itemsize != (Py_ssize_t)sizeof(int32_t) || out->view.len % (Py_ssize_t)sizeof(int32_t) != 0) {
        PyErr_SetString(PyExc_TypeError, "buffer must contain int32 values");
        release_int32_buf(out);
        return 0;
    }
    out->ptr = (const int32_t *)out->view.buf;
    out->n = out->view.len / (Py_ssize_t)sizeof(int32_t);
    return 1;
}

static int solve_alternating_from_scores(
    Py_ssize_t n,
    const double *score0,
    const double *score1,
    const int *fl0,
    const int *fr0,
    const int *fl1,
    const int *fr1,
    int *out_states,
    int *out_lf,
    int *out_rf,
    double *out_total
) {
    if (n < 0) {
        return 0;
    }
    if (n == 0) {
        *out_total = -1e9;
        return 1;
    }

    double total0 = 0.0, total1 = 0.0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        if ((i & 1) == 0) {
            total0 += score0[i];
            total1 += score1[i];
        } else {
            total0 += score1[i];
            total1 += score0[i];
        }
    }

    int start_state = (total0 >= total1) ? 0 : 1;
    *out_total = (start_state == 0) ? total0 : total1;

    for (Py_ssize_t i = 0; i < n; ++i) {
        int state = ((i & 1) == 0) ? start_state : (1 - start_state);
        out_states[i] = state;
        out_lf[i] = (state == 0) ? fl0[i] : fl1[i];
        out_rf[i] = (state == 0) ? fr0[i] : fr1[i];
    }
    return 1;
}

static PyObject *build_solution_tuple(
    Py_ssize_t n,
    const int *states,
    const int *lf,
    const int *rf,
    double total
) {
    PyObject *states_list = PyList_New(n);
    PyObject *left_list = PyList_New(n);
    PyObject *right_list = PyList_New(n);
    if (!states_list || !left_list || !right_list) {
        Py_XDECREF(states_list);
        Py_XDECREF(left_list);
        Py_XDECREF(right_list);
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *py_state = PyLong_FromLong((long)states[i]);
        PyObject *py_l = PyLong_FromLong((long)lf[i]);
        PyObject *py_r = PyLong_FromLong((long)rf[i]);
        if (!py_state || !py_l || !py_r) {
            Py_XDECREF(py_state);
            Py_XDECREF(py_l);
            Py_XDECREF(py_r);
            Py_DECREF(states_list);
            Py_DECREF(left_list);
            Py_DECREF(right_list);
            PyErr_NoMemory();
            return NULL;
        }
        PyList_SET_ITEM(states_list, i, py_state);
        PyList_SET_ITEM(left_list, i, py_l);
        PyList_SET_ITEM(right_list, i, py_r);
    }

    PyObject *ret = Py_BuildValue("OOOd", states_list, left_list, right_list, total);
    Py_DECREF(states_list);
    Py_DECREF(left_list);
    Py_DECREF(right_list);
    return ret;
}

/* Legacy generic solver, keeps list-based API for compatibility. */
static int parse_double_seq(PyObject *obj, double **out, Py_ssize_t *n_out) {
    PyObject *seq = PySequence_Fast(obj, "expected a sequence of floats");
    if (!seq) {
        return 0;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    PyObject **items = PySequence_Fast_ITEMS(seq);

    double *arr = NULL;
    if (n > 0) {
        arr = (double *)PyMem_Malloc((size_t)n * sizeof(double));
        if (!arr) {
            Py_DECREF(seq);
            PyErr_NoMemory();
            return 0;
        }
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
        double v = PyFloat_AsDouble(items[i]);
        if (PyErr_Occurred()) {
            PyMem_Free(arr);
            Py_DECREF(seq);
            return 0;
        }
        arr[i] = v;
    }
    Py_DECREF(seq);
    *out = arr;
    *n_out = n;
    return 1;
}

static int parse_int_seq(PyObject *obj, int **out, Py_ssize_t *n_out) {
    PyObject *seq = PySequence_Fast(obj, "expected a sequence of ints");
    if (!seq) {
        return 0;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    PyObject **items = PySequence_Fast_ITEMS(seq);

    int *arr = NULL;
    if (n > 0) {
        arr = (int *)PyMem_Malloc((size_t)n * sizeof(int));
        if (!arr) {
            Py_DECREF(seq);
            PyErr_NoMemory();
            return 0;
        }
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
        long v = PyLong_AsLong(items[i]);
        if (PyErr_Occurred()) {
            PyMem_Free(arr);
            Py_DECREF(seq);
            return 0;
        }
        arr[i] = (int)v;
    }
    Py_DECREF(seq);
    *out = arr;
    *n_out = n;
    return 1;
}

static PyObject *solve_alternating_path(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *score0_obj, *score1_obj, *fl0_obj, *fr0_obj, *fl1_obj, *fr1_obj;
    if (!PyArg_ParseTuple(args, "OOOOOO", &score0_obj, &score1_obj, &fl0_obj, &fr0_obj, &fl1_obj, &fr1_obj)) {
        return NULL;
    }

    double *score0 = NULL, *score1 = NULL;
    int *fl0 = NULL, *fr0 = NULL, *fl1 = NULL, *fr1 = NULL;
    Py_ssize_t n0 = 0, n1 = 0, nfl0 = 0, nfr0 = 0, nfl1 = 0, nfr1 = 0;

    if (!parse_double_seq(score0_obj, &score0, &n0)) goto fail;
    if (!parse_double_seq(score1_obj, &score1, &n1)) goto fail;
    if (!parse_int_seq(fl0_obj, &fl0, &nfl0)) goto fail;
    if (!parse_int_seq(fr0_obj, &fr0, &nfr0)) goto fail;
    if (!parse_int_seq(fl1_obj, &fl1, &nfl1)) goto fail;
    if (!parse_int_seq(fr1_obj, &fr1, &nfr1)) goto fail;

    if (!(n0 == n1 && n0 == nfl0 && n0 == nfr0 && n0 == nfl1 && n0 == nfr1)) {
        PyErr_SetString(PyExc_ValueError, "all input sequences must have the same length");
        goto fail;
    }

    int *states = NULL, *lf = NULL, *rf = NULL;
    if (n0 > 0) {
        states = (int *)PyMem_Malloc((size_t)n0 * sizeof(int));
        lf = (int *)PyMem_Malloc((size_t)n0 * sizeof(int));
        rf = (int *)PyMem_Malloc((size_t)n0 * sizeof(int));
        if (!states || !lf || !rf) {
            PyErr_NoMemory();
            goto fail;
        }
    }
    double total = -1e9;
    if (!solve_alternating_from_scores(n0, score0, score1, fl0, fr0, fl1, fr1, states, lf, rf, &total)) {
        PyErr_SetString(PyExc_RuntimeError, "failed to solve alternating path");
        goto fail;
    }

    PyObject *ret = build_solution_tuple(n0, states, lf, rf, total);
    PyMem_Free(score0); PyMem_Free(score1); PyMem_Free(fl0); PyMem_Free(fr0); PyMem_Free(fl1); PyMem_Free(fr1);
    PyMem_Free(states); PyMem_Free(lf); PyMem_Free(rf);
    return ret;

fail:
    PyMem_Free(score0); PyMem_Free(score1); PyMem_Free(fl0); PyMem_Free(fr0); PyMem_Free(fl1); PyMem_Free(fr1);
    return NULL;
}

/*
 * solve_bis_intervals
 *
 * Args:
 *   starts[int32], ends[int32],
 *   left_hs[float64], left_to[float64], right_hs[float64], right_to[float64],
 *   hs_mu, hs_sigma, hs_weight, to_mu, to_sigma, to_weight
 *
 * Returns:
 *   (states, left_frames, right_frames, total_score)
 */
static PyObject *solve_bis_intervals(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *starts_obj, *ends_obj;
    PyObject *l_hs_obj, *l_to_obj, *r_hs_obj, *r_to_obj;
    double hs_mu, hs_sigma, hs_weight, to_mu, to_sigma, to_weight;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOOdddddd",
            &starts_obj, &ends_obj,
            &l_hs_obj, &l_to_obj, &r_hs_obj, &r_to_obj,
            &hs_mu, &hs_sigma, &hs_weight,
            &to_mu, &to_sigma, &to_weight)) {
        return NULL;
    }

    Int32Buf starts = {0}, ends = {0};
    DoubleBuf l_hs = {0}, l_to = {0}, r_hs = {0}, r_to = {0};
    if (!parse_int32_buffer(starts_obj, &starts)) goto fail;
    if (!parse_int32_buffer(ends_obj, &ends)) goto fail;
    if (!parse_double_buffer(l_hs_obj, &l_hs)) goto fail;
    if (!parse_double_buffer(l_to_obj, &l_to)) goto fail;
    if (!parse_double_buffer(r_hs_obj, &r_hs)) goto fail;
    if (!parse_double_buffer(r_to_obj, &r_to)) goto fail;

    if (starts.n != ends.n) {
        PyErr_SetString(PyExc_ValueError, "starts and ends must have same length");
        goto fail;
    }
    if (!(l_hs.n == l_to.n && l_hs.n == r_hs.n && l_hs.n == r_to.n)) {
        PyErr_SetString(PyExc_ValueError, "likelihood arrays must have same length");
        goto fail;
    }
    if (hs_sigma <= 0.0 || to_sigma <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "phase sigma must be > 0");
        goto fail;
    }

    Py_ssize_t n_intervals = starts.n;
    int *fl0 = NULL, *fr0 = NULL, *fl1 = NULL, *fr1 = NULL;
    double *score0 = NULL, *score1 = NULL;
    int *states_out = NULL, *lf_out = NULL, *rf_out = NULL;

    if (n_intervals > 0) {
        fl0 = (int *)PyMem_Malloc((size_t)n_intervals * sizeof(int));
        fr0 = (int *)PyMem_Malloc((size_t)n_intervals * sizeof(int));
        fl1 = (int *)PyMem_Malloc((size_t)n_intervals * sizeof(int));
        fr1 = (int *)PyMem_Malloc((size_t)n_intervals * sizeof(int));
        score0 = (double *)PyMem_Malloc((size_t)n_intervals * sizeof(double));
        score1 = (double *)PyMem_Malloc((size_t)n_intervals * sizeof(double));
        states_out = (int *)PyMem_Malloc((size_t)n_intervals * sizeof(int));
        lf_out = (int *)PyMem_Malloc((size_t)n_intervals * sizeof(int));
        rf_out = (int *)PyMem_Malloc((size_t)n_intervals * sizeof(int));
        if (!fl0 || !fr0 || !fl1 || !fr1 || !score0 || !score1 || !states_out || !lf_out || !rf_out) {
            PyErr_NoMemory();
            goto fail_alloc;
        }
    }

    Py_ssize_t n_frames = l_hs.n;
    for (Py_ssize_t i = 0; i < n_intervals; ++i) {
        int32_t s = starts.ptr[i];
        int32_t e = ends.ptr[i];

        if (s < 0) s = 0;
        if (e < 0) e = 0;
        if (s > (int32_t)n_frames) s = (int32_t)n_frames;
        if (e > (int32_t)n_frames) e = (int32_t)n_frames;

        if (e <= s) {
            score0[i] = -1e9;
            score1[i] = -1e9;
            fl0[i] = fr0[i] = fl1[i] = fr1[i] = (int)s;
            continue;
        }

        int32_t L = e - s;
        double best_lt = -1e300, best_rh = -1e300, best_lh = -1e300, best_rt = -1e300;
        int best_lt_f = (int)s, best_rh_f = (int)s, best_lh_f = (int)s, best_rt_f = (int)s;

        for (int32_t k = 0; k < L; ++k) {
            double phase = (L > 1) ? ((double)k / (double)(L - 1)) : 0.0;
            double z_hs = (phase - hs_mu) / hs_sigma;
            double z_to = (phase - to_mu) / to_sigma;
            double hs_phase = hs_weight * (-0.5 * z_hs * z_hs);
            double to_phase = to_weight * (-0.5 * z_to * z_to);
            Py_ssize_t idx = (Py_ssize_t)s + (Py_ssize_t)k;

            double v_lt = l_to.ptr[idx] + to_phase;
            double v_rh = r_hs.ptr[idx] + hs_phase;
            double v_lh = l_hs.ptr[idx] + hs_phase;
            double v_rt = r_to.ptr[idx] + to_phase;

            if (v_lt > best_lt) { best_lt = v_lt; best_lt_f = (int)idx; }
            if (v_rh > best_rh) { best_rh = v_rh; best_rh_f = (int)idx; }
            if (v_lh > best_lh) { best_lh = v_lh; best_lh_f = (int)idx; }
            if (v_rt > best_rt) { best_rt = v_rt; best_rt_f = (int)idx; }
        }

        score0[i] = best_lt + best_rh; /* (TO, HS) */
        score1[i] = best_lh + best_rt; /* (HS, TO) */
        fl0[i] = best_lt_f; fr0[i] = best_rh_f;
        fl1[i] = best_lh_f; fr1[i] = best_rt_f;
    }

    double total = -1e9;
    if (!solve_alternating_from_scores(n_intervals, score0, score1, fl0, fr0, fl1, fr1, states_out, lf_out, rf_out, &total)) {
        PyErr_SetString(PyExc_RuntimeError, "failed to solve intervals");
        goto fail_alloc;
    }

    {
        PyObject *ret = build_solution_tuple(n_intervals, states_out, lf_out, rf_out, total);
        PyMem_Free(fl0); PyMem_Free(fr0); PyMem_Free(fl1); PyMem_Free(fr1);
        PyMem_Free(score0); PyMem_Free(score1); PyMem_Free(states_out); PyMem_Free(lf_out); PyMem_Free(rf_out);
        release_int32_buf(&starts); release_int32_buf(&ends);
        release_double_buf(&l_hs); release_double_buf(&l_to); release_double_buf(&r_hs); release_double_buf(&r_to);
        return ret;
    }

fail_alloc:
    PyMem_Free(fl0); PyMem_Free(fr0); PyMem_Free(fl1); PyMem_Free(fr1);
    PyMem_Free(score0); PyMem_Free(score1); PyMem_Free(states_out); PyMem_Free(lf_out); PyMem_Free(rf_out);
fail:
    release_int32_buf(&starts); release_int32_buf(&ends);
    release_double_buf(&l_hs); release_double_buf(&l_to); release_double_buf(&r_hs); release_double_buf(&r_to);
    return NULL;
}

/*
 * solve_prob_intervals
 *
 * Args:
 *   starts[int32], ends[int32],
 *   left_hs, left_to, right_hs, right_to (probabilities in [0,1])
 *
 * Returns:
 *   (states, left_frames, right_frames, total_score)
 */
static PyObject *solve_prob_intervals(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *starts_obj, *ends_obj;
    PyObject *l_hs_obj, *l_to_obj, *r_hs_obj, *r_to_obj;
    if (!PyArg_ParseTuple(args, "OOOOOO", &starts_obj, &ends_obj, &l_hs_obj, &l_to_obj, &r_hs_obj, &r_to_obj)) {
        return NULL;
    }

    Int32Buf starts = {0}, ends = {0};
    DoubleBuf l_hs = {0}, l_to = {0}, r_hs = {0}, r_to = {0};
    if (!parse_int32_buffer(starts_obj, &starts)) goto fail;
    if (!parse_int32_buffer(ends_obj, &ends)) goto fail;
    if (!parse_double_buffer(l_hs_obj, &l_hs)) goto fail;
    if (!parse_double_buffer(l_to_obj, &l_to)) goto fail;
    if (!parse_double_buffer(r_hs_obj, &r_hs)) goto fail;
    if (!parse_double_buffer(r_to_obj, &r_to)) goto fail;

    if (starts.n != ends.n) {
        PyErr_SetString(PyExc_ValueError, "starts and ends must have same length");
        goto fail;
    }
    if (!(l_hs.n == l_to.n && l_hs.n == r_hs.n && l_hs.n == r_to.n)) {
        PyErr_SetString(PyExc_ValueError, "prob arrays must have same length");
        goto fail;
    }

    Py_ssize_t n_intervals = starts.n;
    Py_ssize_t n_frames = l_hs.n;
    int *fl0 = NULL, *fr0 = NULL, *fl1 = NULL, *fr1 = NULL;
    double *score0 = NULL, *score1 = NULL;
    int *states_out = NULL, *lf_out = NULL, *rf_out = NULL;

    if (n_intervals > 0) {
        fl0 = (int *)PyMem_Malloc((size_t)n_intervals * sizeof(int));
        fr0 = (int *)PyMem_Malloc((size_t)n_intervals * sizeof(int));
        fl1 = (int *)PyMem_Malloc((size_t)n_intervals * sizeof(int));
        fr1 = (int *)PyMem_Malloc((size_t)n_intervals * sizeof(int));
        score0 = (double *)PyMem_Malloc((size_t)n_intervals * sizeof(double));
        score1 = (double *)PyMem_Malloc((size_t)n_intervals * sizeof(double));
        states_out = (int *)PyMem_Malloc((size_t)n_intervals * sizeof(int));
        lf_out = (int *)PyMem_Malloc((size_t)n_intervals * sizeof(int));
        rf_out = (int *)PyMem_Malloc((size_t)n_intervals * sizeof(int));
        if (!fl0 || !fr0 || !fl1 || !fr1 || !score0 || !score1 || !states_out || !lf_out || !rf_out) {
            PyErr_NoMemory();
            goto fail_alloc;
        }
    }

    for (Py_ssize_t i = 0; i < n_intervals; ++i) {
        int32_t s = starts.ptr[i];
        int32_t e = ends.ptr[i];

        if (s < 0) s = 0;
        if (e < 0) e = 0;
        if (s > (int32_t)n_frames) s = (int32_t)n_frames;
        if (e > (int32_t)n_frames) e = (int32_t)n_frames;

        if (e <= s) {
            score0[i] = -1e9;
            score1[i] = -1e9;
            fl0[i] = fr0[i] = fl1[i] = fr1[i] = (int)s;
            continue;
        }

        double best_lt = -1.0, best_rh = -1.0, best_lh = -1.0, best_rt = -1.0;
        int best_lt_f = (int)s, best_rh_f = (int)s, best_lh_f = (int)s, best_rt_f = (int)s;
        for (int32_t k = s; k < e; ++k) {
            double v_lt = l_to.ptr[k];
            double v_rh = r_hs.ptr[k];
            double v_lh = l_hs.ptr[k];
            double v_rt = r_to.ptr[k];
            if (v_lt > best_lt) { best_lt = v_lt; best_lt_f = (int)k; }
            if (v_rh > best_rh) { best_rh = v_rh; best_rh_f = (int)k; }
            if (v_lh > best_lh) { best_lh = v_lh; best_lh_f = (int)k; }
            if (v_rt > best_rt) { best_rt = v_rt; best_rt_f = (int)k; }
        }

        score0[i] = log(best_lt + 1e-10) + log(best_rh + 1e-10);
        score1[i] = log(best_lh + 1e-10) + log(best_rt + 1e-10);
        fl0[i] = best_lt_f; fr0[i] = best_rh_f;
        fl1[i] = best_lh_f; fr1[i] = best_rt_f;
    }

    double total = -1e9;
    if (!solve_alternating_from_scores(n_intervals, score0, score1, fl0, fr0, fl1, fr1, states_out, lf_out, rf_out, &total)) {
        PyErr_SetString(PyExc_RuntimeError, "failed to solve intervals");
        goto fail_alloc;
    }

    {
        PyObject *ret = build_solution_tuple(n_intervals, states_out, lf_out, rf_out, total);
        PyMem_Free(fl0); PyMem_Free(fr0); PyMem_Free(fl1); PyMem_Free(fr1);
        PyMem_Free(score0); PyMem_Free(score1); PyMem_Free(states_out); PyMem_Free(lf_out); PyMem_Free(rf_out);
        release_int32_buf(&starts); release_int32_buf(&ends);
        release_double_buf(&l_hs); release_double_buf(&l_to); release_double_buf(&r_hs); release_double_buf(&r_to);
        return ret;
    }

fail_alloc:
    PyMem_Free(fl0); PyMem_Free(fr0); PyMem_Free(fl1); PyMem_Free(fr1);
    PyMem_Free(score0); PyMem_Free(score1); PyMem_Free(states_out); PyMem_Free(lf_out); PyMem_Free(rf_out);
fail:
    release_int32_buf(&starts); release_int32_buf(&ends);
    release_double_buf(&l_hs); release_double_buf(&l_to); release_double_buf(&r_hs); release_double_buf(&r_to);
    return NULL;
}

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static double median_sorted(const double *arr, Py_ssize_t n) {
    if (n <= 0) return 0.0;
    if (n & 1) return arr[n / 2];
    return 0.5 * (arr[(n / 2) - 1] + arr[n / 2]);
}

/*
 * calc_p_signal_raw
 *
 * Args:
 *   diff[float64], dv[float64], min_crossing_frames[int],
 *   mad_to_sigma_factor, local_blend_local, local_blend_global,
 *   local_sigma_min_ratio, proximity_bandwidth
 *
 * Returns:
 *   ps_raw as Python list[float], normalized to max=1 before smoothing.
 */
static PyObject *calc_p_signal_raw(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *diff_obj, *dv_obj;
    int min_crossing_frames;
    double mad_to_sigma_factor, local_blend_local, local_blend_global;
    double local_sigma_min_ratio, proximity_bandwidth;

    if (!PyArg_ParseTuple(
            args,
            "OOiddddd",
            &diff_obj, &dv_obj, &min_crossing_frames,
            &mad_to_sigma_factor, &local_blend_local, &local_blend_global,
            &local_sigma_min_ratio, &proximity_bandwidth)) {
        return NULL;
    }

    DoubleBuf diff = {0}, dv = {0};
    if (!parse_double_buffer(diff_obj, &diff)) goto fail;
    if (!parse_double_buffer(dv_obj, &dv)) goto fail;
    if (diff.n != dv.n) {
        PyErr_SetString(PyExc_ValueError, "diff and dv must have same length");
        goto fail;
    }

    Py_ssize_t n = diff.n;
    PyObject *out = PyList_New(n);
    if (!out) {
        PyErr_NoMemory();
        goto fail;
    }
    if (n == 0) {
        release_double_buf(&diff);
        release_double_buf(&dv);
        return out;
    }

    /* ds_global = std(diff) with ddof=0 */
    double sum = 0.0;
    for (Py_ssize_t i = 0; i < n; ++i) sum += diff.ptr[i];
    double mean = sum / (double)n;
    double var = 0.0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        double d = diff.ptr[i] - mean;
        var += d * d;
    }
    var /= (double)n;
    double ds_global = sqrt(var);

    double *ps = (double *)PyMem_Malloc((size_t)n * sizeof(double));
    double *va = (double *)PyMem_Malloc((size_t)n * sizeof(double));
    if (!ps || !va) {
        PyMem_Free(ps);
        PyMem_Free(va);
        Py_DECREF(out);
        PyErr_NoMemory();
        goto fail;
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
        va[i] = fabs(dv.ptr[i]);
    }

    if (n > 20 && ds_global > 0.0) {
        int half_win = (min_crossing_frames * 2 > 30) ? (min_crossing_frames * 2) : 30;
        int vel_half_win = (min_crossing_frames > 15) ? min_crossing_frames : 15;
        Py_ssize_t maxw = (Py_ssize_t)(2 * half_win + 1);
        if (maxw > n) maxw = n;

        double *tmp = (double *)PyMem_Malloc((size_t)maxw * sizeof(double));
        double *tmp_abs = (double *)PyMem_Malloc((size_t)maxw * sizeof(double));
        if (!tmp || !tmp_abs) {
            PyMem_Free(tmp);
            PyMem_Free(tmp_abs);
            PyMem_Free(ps);
            PyMem_Free(va);
            Py_DECREF(out);
            PyErr_NoMemory();
            goto fail;
        }

        for (Py_ssize_t i = 0; i < n; ++i) {
            Py_ssize_t ws = i - half_win;
            Py_ssize_t we = i + half_win + 1;
            if (ws < 0) ws = 0;
            if (we > n) we = n;
            Py_ssize_t L = we - ws;

            double sigma_local = ds_global;
            if (L > 10) {
                for (Py_ssize_t j = 0; j < L; ++j) tmp[j] = diff.ptr[ws + j];
                qsort(tmp, (size_t)L, sizeof(double), cmp_double);
                double med = median_sorted(tmp, L);
                for (Py_ssize_t j = 0; j < L; ++j) {
                    tmp_abs[j] = fabs(diff.ptr[ws + j] - med);
                }
                qsort(tmp_abs, (size_t)L, sizeof(double), cmp_double);
                double mad = median_sorted(tmp_abs, L);
                double sigma_mad = mad_to_sigma_factor * mad;
                if (sigma_mad > local_sigma_min_ratio * ds_global) {
                    sigma_local = local_blend_local * sigma_mad + local_blend_global * ds_global;
                }
            }

            double sigma_kernel = sigma_local * proximity_bandwidth;
            if (sigma_kernel < 0.01) sigma_kernel = 0.01;
            double ratio = diff.ptr[i] / sigma_kernel;
            double pp = exp(-0.5 * ratio * ratio);

            Py_ssize_t vws = i - vel_half_win;
            Py_ssize_t vwe = i + vel_half_win + 1;
            if (vws < 0) vws = 0;
            if (vwe > n) vwe = n;
            double local_max = 0.0;
            for (Py_ssize_t j = vws; j < vwe; ++j) {
                if (va[j] > local_max) local_max = va[j];
            }
            if (local_max < 1e-10) local_max = 1e-10;
            ps[i] = pp * (va[i] / local_max);
        }
        PyMem_Free(tmp);
        PyMem_Free(tmp_abs);
    } else {
        double vm = 0.0;
        for (Py_ssize_t i = 0; i < n; ++i) if (va[i] > vm) vm = va[i];
        if (vm <= 0.0) vm = 1.0;
        for (Py_ssize_t i = 0; i < n; ++i) {
            double pp;
            if (ds_global > 0.0) {
                double denom = ds_global * proximity_bandwidth;
                double ratio = diff.ptr[i] / denom;
                pp = exp(-0.5 * ratio * ratio);
            } else {
                pp = 1.0;
            }
            ps[i] = pp * (va[i] / vm);
        }
    }

    double pm = 0.0;
    for (Py_ssize_t i = 0; i < n; ++i) if (ps[i] > pm) pm = ps[i];
    if (pm > 0.0) {
        for (Py_ssize_t i = 0; i < n; ++i) ps[i] /= pm;
    }

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *v = PyFloat_FromDouble(ps[i]);
        if (!v) {
            PyMem_Free(ps);
            PyMem_Free(va);
            Py_DECREF(out);
            PyErr_NoMemory();
            goto fail;
        }
        PyList_SET_ITEM(out, i, v);
    }

    PyMem_Free(ps);
    PyMem_Free(va);
    release_double_buf(&diff);
    release_double_buf(&dv);
    return out;

fail:
    release_double_buf(&diff);
    release_double_buf(&dv);
    return NULL;
}

/*
 * compute_dgei_curves_raw
 *
 * Args:
 *   signal[float64], fps[float], bar_threshold[float]
 *
 * Returns:
 *   (dgei_pos_raw list[float], dgei_neg_raw list[float])
 */
static PyObject *compute_dgei_curves_raw(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *signal_obj;
    double fps, bar_threshold;
    if (!PyArg_ParseTuple(args, "Odd", &signal_obj, &fps, &bar_threshold)) {
        return NULL;
    }

    DoubleBuf signal = {0};
    if (!parse_double_buffer(signal_obj, &signal)) {
        return NULL;
    }

    Py_ssize_t n = signal.n;
    PyObject *pos_list = PyList_New(n);
    PyObject *neg_list = PyList_New(n);
    if (!pos_list || !neg_list) {
        Py_XDECREF(pos_list);
        Py_XDECREF(neg_list);
        release_double_buf(&signal);
        PyErr_NoMemory();
        return NULL;
    }

    if (n == 0) {
        release_double_buf(&signal);
        return Py_BuildValue("OO", pos_list, neg_list);
    }

    double *dy = (double *)PyMem_Malloc((size_t)n * sizeof(double));
    double *pos = (double *)PyMem_Malloc((size_t)n * sizeof(double));
    double *neg = (double *)PyMem_Malloc((size_t)n * sizeof(double));
    if (!dy || !pos || !neg) {
        PyMem_Free(dy); PyMem_Free(pos); PyMem_Free(neg);
        Py_DECREF(pos_list); Py_DECREF(neg_list);
        release_double_buf(&signal);
        PyErr_NoMemory();
        return NULL;
    }

    dy[0] = 0.0;
    pos[0] = 0.0;
    neg[0] = 0.0;
    for (Py_ssize_t i = 1; i < n; ++i) {
        dy[i] = signal.ptr[i] - signal.ptr[i - 1];
        pos[i] = 0.0;
        neg[i] = 0.0;
    }

    /* sigma_dy = std(dy), mu_dv = mean(abs(dy)) */
    double mean_dy = 0.0;
    for (Py_ssize_t i = 0; i < n; ++i) mean_dy += dy[i];
    mean_dy /= (double)n;

    double var_dy = 0.0;
    double mean_abs = 0.0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        double d = dy[i] - mean_dy;
        var_dy += d * d;
        mean_abs += fabs(dy[i]);
    }
    var_dy /= (double)n;
    mean_abs /= (double)n;
    double sigma_dy = sqrt(var_dy) + 1e-10;
    double mu_dv = mean_abs + 1e-10;
    double alpha = sigma_dy / (sigma_dy + mu_dv);
    double beta = mu_dv / (sigma_dy + mu_dv);

    int window_size = (int)(0.05 * fps);
    if (window_size < 3) window_size = 3;

    for (Py_ssize_t i = window_size; i < n; ++i) {
        Py_ssize_t ws = i - window_size;
        double sp = 0.0;
        double sn = 0.0;
        for (Py_ssize_t j = ws; j < i; ++j) {
            double v = dy[j];
            if (v > bar_threshold) {
                sp += (alpha * v + beta * v);
            }
            if (v < -bar_threshold) {
                double av = fabs(v);
                sn += (alpha * av + beta * av);
            }
        }
        pos[i] = sp;
        neg[i] = sn;
    }

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *pv = PyFloat_FromDouble(pos[i]);
        PyObject *nv = PyFloat_FromDouble(neg[i]);
        if (!pv || !nv) {
            Py_XDECREF(pv);
            Py_XDECREF(nv);
            PyMem_Free(dy); PyMem_Free(pos); PyMem_Free(neg);
            Py_DECREF(pos_list); Py_DECREF(neg_list);
            release_double_buf(&signal);
            PyErr_NoMemory();
            return NULL;
        }
        PyList_SET_ITEM(pos_list, i, pv);
        PyList_SET_ITEM(neg_list, i, nv);
    }

    PyMem_Free(dy); PyMem_Free(pos); PyMem_Free(neg);
    release_double_buf(&signal);
    return Py_BuildValue("OO", pos_list, neg_list);
}

/*
 * zero_crossings_neg_to_pos
 *
 * Args:
 *   vel[float64], min_interval[int]
 *
 * Returns:
 *   list[int] crossings filtered with minimum frame distance.
 */
static PyObject *zero_crossings_neg_to_pos(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *vel_obj;
    int min_interval;
    if (!PyArg_ParseTuple(args, "Oi", &vel_obj, &min_interval)) {
        return NULL;
    }
    if (min_interval < 1) min_interval = 1;

    DoubleBuf vel = {0};
    if (!parse_double_buffer(vel_obj, &vel)) {
        return NULL;
    }

    Py_ssize_t n = vel.n;
    int *events = NULL;
    if (n > 0) {
        events = (int *)PyMem_Malloc((size_t)n * sizeof(int));
        if (!events) {
            release_double_buf(&vel);
            PyErr_NoMemory();
            return NULL;
        }
    }

    Py_ssize_t count = 0;
    for (Py_ssize_t i = 1; i < n; ++i) {
        if (vel.ptr[i - 1] < 0.0 && vel.ptr[i] >= 0.0) {
            int fi = (int)i;
            if (count == 0 || (fi - events[count - 1]) >= min_interval) {
                events[count++] = fi;
            }
        }
    }

    PyObject *out = PyList_New(count);
    if (!out) {
        PyMem_Free(events);
        release_double_buf(&vel);
        PyErr_NoMemory();
        return NULL;
    }
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject *v = PyLong_FromLong((long)events[i]);
        if (!v) {
            Py_DECREF(out);
            PyMem_Free(events);
            release_double_buf(&vel);
            PyErr_NoMemory();
            return NULL;
        }
        PyList_SET_ITEM(out, i, v);
    }

    PyMem_Free(events);
    release_double_buf(&vel);
    return out;
}

/*
 * mickelborough_events_raw
 *
 * Args:
 *   vel[float64], threshold_fraction[float], min_interval[int]
 *
 * Returns:
 *   (hs_frames list[int], to_frames list[int])
 */
static PyObject *mickelborough_events_raw(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *vel_obj;
    double threshold_fraction;
    int min_interval;
    if (!PyArg_ParseTuple(args, "Odi", &vel_obj, &threshold_fraction, &min_interval)) {
        return NULL;
    }
    if (min_interval < 1) min_interval = 1;

    DoubleBuf vel = {0};
    if (!parse_double_buffer(vel_obj, &vel)) {
        return NULL;
    }

    Py_ssize_t n = vel.n;
    int *hs = NULL, *to = NULL;
    if (n > 0) {
        hs = (int *)PyMem_Malloc((size_t)n * sizeof(int));
        to = (int *)PyMem_Malloc((size_t)n * sizeof(int));
        if (!hs || !to) {
            PyMem_Free(hs);
            PyMem_Free(to);
            release_double_buf(&vel);
            PyErr_NoMemory();
            return NULL;
        }
    }

    double vmin = 0.0, vmax = 0.0;
    if (n > 0) {
        vmin = vel.ptr[0];
        vmax = vel.ptr[0];
    }
    for (Py_ssize_t i = 1; i < n; ++i) {
        double v = vel.ptr[i];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }
    double threshold = threshold_fraction * (vmax - vmin);

    Py_ssize_t hs_count = 0, to_count = 0;
    for (Py_ssize_t i = 1; i < n; ++i) {
        double prev = vel.ptr[i - 1];
        double cur = vel.ptr[i];
        int fi = (int)i;

        if (prev < -threshold && cur >= -threshold) {
            if (hs_count == 0 || (fi - hs[hs_count - 1]) >= min_interval) {
                hs[hs_count++] = fi;
            }
        } else if (prev <= threshold && cur > threshold) {
            if (to_count == 0 || (fi - to[to_count - 1]) >= min_interval) {
                to[to_count++] = fi;
            }
        }
    }

    PyObject *hs_list = PyList_New(hs_count);
    PyObject *to_list = PyList_New(to_count);
    if (!hs_list || !to_list) {
        Py_XDECREF(hs_list);
        Py_XDECREF(to_list);
        PyMem_Free(hs);
        PyMem_Free(to);
        release_double_buf(&vel);
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < hs_count; ++i) {
        PyObject *v = PyLong_FromLong((long)hs[i]);
        if (!v) {
            Py_DECREF(hs_list);
            Py_DECREF(to_list);
            PyMem_Free(hs);
            PyMem_Free(to);
            release_double_buf(&vel);
            PyErr_NoMemory();
            return NULL;
        }
        PyList_SET_ITEM(hs_list, i, v);
    }
    for (Py_ssize_t i = 0; i < to_count; ++i) {
        PyObject *v = PyLong_FromLong((long)to[i]);
        if (!v) {
            Py_DECREF(hs_list);
            Py_DECREF(to_list);
            PyMem_Free(hs);
            PyMem_Free(to);
            release_double_buf(&vel);
            PyErr_NoMemory();
            return NULL;
        }
        PyList_SET_ITEM(to_list, i, v);
    }

    PyMem_Free(hs);
    PyMem_Free(to);
    release_double_buf(&vel);
    return Py_BuildValue("OO", hs_list, to_list);
}

static const double *g_peak_priority = NULL;
static const int *g_peak_indices = NULL;

static int cmp_peak_priority_desc(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    double pa = g_peak_priority[ia];
    double pb = g_peak_priority[ib];
    if (pa > pb) return -1;
    if (pa < pb) return 1;
    if (g_peak_indices[ia] < g_peak_indices[ib]) return -1;
    if (g_peak_indices[ia] > g_peak_indices[ib]) return 1;
    return 0;
}

static int cmp_int_asc(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    if (ia < ib) return -1;
    if (ia > ib) return 1;
    return 0;
}

static Py_ssize_t find_peaks_core(
    const double *x,
    Py_ssize_t n,
    int distance,
    double prominence,
    int mode,
    int *out_idx
) {
    if (distance < 1) distance = 1;
    if (n < 3) return 0;

    int *cand_idx = (int *)PyMem_Malloc((size_t)n * sizeof(int));
    double *cand_prio = (double *)PyMem_Malloc((size_t)n * sizeof(double));
    if (!cand_idx || !cand_prio) {
        PyMem_Free(cand_idx);
        PyMem_Free(cand_prio);
        return -1;
    }

    Py_ssize_t n_cand = 0;
    for (Py_ssize_t i = 1; i + 1 < n; ++i) {
        double ym1 = mode * x[i - 1];
        double y0 = mode * x[i];
        double yp1 = mode * x[i + 1];
        if (!(y0 > ym1 && y0 >= yp1)) {
            continue;
        }

        double left_min = y0;
        for (Py_ssize_t j = i; j > 0; --j) {
            double y = mode * x[j - 1];
            if (y < left_min) left_min = y;
            if (y > y0) break;
        }

        double right_min = y0;
        for (Py_ssize_t j = i; j + 1 < n; ++j) {
            double y = mode * x[j + 1];
            if (y < right_min) right_min = y;
            if (y > y0) break;
        }

        double base = (left_min > right_min) ? left_min : right_min;
        if ((y0 - base) < prominence) {
            continue;
        }

        cand_idx[n_cand] = (int)i;
        cand_prio[n_cand] = y0;
        n_cand += 1;
    }

    if (n_cand <= 0) {
        PyMem_Free(cand_idx);
        PyMem_Free(cand_prio);
        return 0;
    }

    if (distance <= 1) {
        for (Py_ssize_t i = 0; i < n_cand; ++i) {
            out_idx[i] = cand_idx[i];
        }
        PyMem_Free(cand_idx);
        PyMem_Free(cand_prio);
        return n_cand;
    }

    int *order = (int *)PyMem_Malloc((size_t)n_cand * sizeof(int));
    unsigned char *keep = (unsigned char *)PyMem_Malloc((size_t)n_cand * sizeof(unsigned char));
    if (!order || !keep) {
        PyMem_Free(order);
        PyMem_Free(keep);
        PyMem_Free(cand_idx);
        PyMem_Free(cand_prio);
        return -1;
    }

    for (Py_ssize_t i = 0; i < n_cand; ++i) {
        order[i] = (int)i;
        keep[i] = 1;
    }

    g_peak_priority = cand_prio;
    g_peak_indices = cand_idx;
    qsort(order, (size_t)n_cand, sizeof(int), cmp_peak_priority_desc);
    g_peak_priority = NULL;
    g_peak_indices = NULL;

    for (Py_ssize_t oi = 0; oi < n_cand; ++oi) {
        int ci = order[oi];
        if (!keep[ci]) continue;
        int peak_i = cand_idx[ci];
        for (Py_ssize_t oj = oi + 1; oj < n_cand; ++oj) {
            int cj = order[oj];
            if (!keep[cj]) continue;
            int d = cand_idx[cj] - peak_i;
            if (d < 0) d = -d;
            if (d < distance) {
                keep[cj] = 0;
            }
        }
    }

    Py_ssize_t out_count = 0;
    for (Py_ssize_t i = 0; i < n_cand; ++i) {
        if (keep[i]) {
            out_idx[out_count++] = cand_idx[i];
        }
    }
    if (out_count > 1) {
        qsort(out_idx, (size_t)out_count, sizeof(int), cmp_int_asc);
    }

    PyMem_Free(order);
    PyMem_Free(keep);
    PyMem_Free(cand_idx);
    PyMem_Free(cand_prio);
    return out_count;
}

/*
 * find_peaks_filtered
 *
 * Args:
 *   signal[float64], distance[int], prominence[float], mode[int]
 *
 * Returns:
 *   list[int] peak indices.
 *
 * Notes:
 *   mode=+1 for maxima, mode=-1 for minima.
 */
static PyObject *find_peaks_filtered(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *sig_obj;
    int distance, mode;
    double prominence;
    if (!PyArg_ParseTuple(args, "Oidi", &sig_obj, &distance, &prominence, &mode)) {
        return NULL;
    }
    if (mode != 1 && mode != -1) {
        PyErr_SetString(PyExc_ValueError, "mode must be +1 (max) or -1 (min)");
        return NULL;
    }
    if (distance < 1) distance = 1;
    if (prominence < 0.0) prominence = 0.0;

    DoubleBuf sig = {0};
    if (!parse_double_buffer(sig_obj, &sig)) {
        return NULL;
    }

    Py_ssize_t n = sig.n;
    int *tmp = NULL;
    if (n > 0) {
        tmp = (int *)PyMem_Malloc((size_t)n * sizeof(int));
        if (!tmp) {
            release_double_buf(&sig);
            PyErr_NoMemory();
            return NULL;
        }
    }

    Py_ssize_t count = find_peaks_core(sig.ptr, n, distance, prominence, mode, tmp);
    if (count < 0) {
        PyMem_Free(tmp);
        release_double_buf(&sig);
        PyErr_NoMemory();
        return NULL;
    }
    PyObject *out = PyList_New(count);
    if (!out) {
        PyMem_Free(tmp);
        release_double_buf(&sig);
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject *v = PyLong_FromLong((long)tmp[i]);
        if (!v) {
            Py_DECREF(out);
            PyMem_Free(tmp);
            release_double_buf(&sig);
            PyErr_NoMemory();
            return NULL;
        }
        PyList_SET_ITEM(out, i, v);
    }

    PyMem_Free(tmp);
    release_double_buf(&sig);
    return out;
}

static PyMethodDef GaitNativeMethods[] = {
    {
        "solve_alternating_path",
        solve_alternating_path,
        METH_VARARGS,
        "Solve constrained 2-state alternating path from precomputed scores."
    },
    {
        "solve_bis_intervals",
        solve_bis_intervals,
        METH_VARARGS,
        "Fast Bayesian BIS interval assignment with phase priors."
    },
    {
        "solve_prob_intervals",
        solve_prob_intervals,
        METH_VARARGS,
        "Fast fallback interval assignment from event probabilities."
    },
    {
        "calc_p_signal_raw",
        calc_p_signal_raw,
        METH_VARARGS,
        "Compute raw normalized p-signal before Gaussian smoothing."
    },
    {
        "compute_dgei_curves_raw",
        compute_dgei_curves_raw,
        METH_VARARGS,
        "Compute raw DGEI positive/negative curves before smoothing."
    },
    {
        "zero_crossings_neg_to_pos",
        zero_crossings_neg_to_pos,
        METH_VARARGS,
        "Find negative-to-positive zero crossings with min distance filtering."
    },
    {
        "mickelborough_events_raw",
        mickelborough_events_raw,
        METH_VARARGS,
        "Detect Mickelborough HS/TO threshold crossings from vertical velocity."
    },
    {
        "find_peaks_filtered",
        find_peaks_filtered,
        METH_VARARGS,
        "Find local peaks with distance/prominence filtering (mode=+1/-1)."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef gait_native_module = {
    PyModuleDef_HEAD_INIT,
    "_gait_native",
    "Native helpers for gait event detection.",
    -1,
    GaitNativeMethods
};

PyMODINIT_FUNC PyInit__gait_native(void) {
    return PyModule_Create(&gait_native_module);
}
