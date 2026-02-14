from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.signal import find_peaks

from recode.native import _gait_native


def _solve_alternating_reference(score0, score1, fl0, fr0, fl1, fr1):
    n = len(score0)
    if n == 0:
        return [], [], [], -1e9
    total0 = 0.0
    total1 = 0.0
    for i in range(n):
        if (i % 2) == 0:
            total0 += score0[i]
            total1 += score1[i]
        else:
            total0 += score1[i]
            total1 += score0[i]
    start = 0 if total0 >= total1 else 1
    states = [start if (i % 2) == 0 else 1 - start for i in range(n)]
    lf = [fl0[i] if st == 0 else fl1[i] for i, st in enumerate(states)]
    rf = [fr0[i] if st == 0 else fr1[i] for i, st in enumerate(states)]
    total = float(total0 if start == 0 else total1)
    return states, lf, rf, total


def _solve_prob_intervals_reference(starts, ends, l_hs, l_to, r_hs, r_to):
    n_intervals = len(starts)
    n_frames = len(l_hs)
    fl0 = [0] * n_intervals
    fr0 = [0] * n_intervals
    fl1 = [0] * n_intervals
    fr1 = [0] * n_intervals
    score0 = [0.0] * n_intervals
    score1 = [0.0] * n_intervals

    for i, (s0, e0) in enumerate(zip(starts, ends)):
        s = max(0, min(int(s0), n_frames))
        e = max(0, min(int(e0), n_frames))
        if e <= s:
            score0[i] = -1e9
            score1[i] = -1e9
            fl0[i] = fr0[i] = fl1[i] = fr1[i] = s
            continue

        seg = slice(s, e)
        lt_seg = l_to[seg]
        rh_seg = r_hs[seg]
        lh_seg = l_hs[seg]
        rt_seg = r_to[seg]

        k_lt = int(np.argmax(lt_seg))
        k_rh = int(np.argmax(rh_seg))
        k_lh = int(np.argmax(lh_seg))
        k_rt = int(np.argmax(rt_seg))

        best_lt = float(lt_seg[k_lt])
        best_rh = float(rh_seg[k_rh])
        best_lh = float(lh_seg[k_lh])
        best_rt = float(rt_seg[k_rt])

        fl0[i] = s + k_lt
        fr0[i] = s + k_rh
        fl1[i] = s + k_lh
        fr1[i] = s + k_rt
        score0[i] = math.log(best_lt + 1e-10) + math.log(best_rh + 1e-10)
        score1[i] = math.log(best_lh + 1e-10) + math.log(best_rt + 1e-10)

    return _solve_alternating_reference(score0, score1, fl0, fr0, fl1, fr1)


def _solve_bis_intervals_reference(
    starts,
    ends,
    l_hs,
    l_to,
    r_hs,
    r_to,
    hs_mu,
    hs_sigma,
    hs_weight,
    to_mu,
    to_sigma,
    to_weight,
):
    n_intervals = len(starts)
    n_frames = len(l_hs)
    fl0 = [0] * n_intervals
    fr0 = [0] * n_intervals
    fl1 = [0] * n_intervals
    fr1 = [0] * n_intervals
    score0 = [0.0] * n_intervals
    score1 = [0.0] * n_intervals

    for i, (s0, e0) in enumerate(zip(starts, ends)):
        s = max(0, min(int(s0), n_frames))
        e = max(0, min(int(e0), n_frames))
        if e <= s:
            score0[i] = -1e9
            score1[i] = -1e9
            fl0[i] = fr0[i] = fl1[i] = fr1[i] = s
            continue

        L = e - s
        best_lt = -1e300
        best_rh = -1e300
        best_lh = -1e300
        best_rt = -1e300
        best_lt_f = s
        best_rh_f = s
        best_lh_f = s
        best_rt_f = s

        for k in range(L):
            phase = (k / (L - 1)) if L > 1 else 0.0
            z_hs = (phase - hs_mu) / hs_sigma
            z_to = (phase - to_mu) / to_sigma
            hs_phase = hs_weight * (-0.5 * z_hs * z_hs)
            to_phase = to_weight * (-0.5 * z_to * z_to)
            idx = s + k

            v_lt = float(l_to[idx]) + to_phase
            v_rh = float(r_hs[idx]) + hs_phase
            v_lh = float(l_hs[idx]) + hs_phase
            v_rt = float(r_to[idx]) + to_phase

            if v_lt > best_lt:
                best_lt = v_lt
                best_lt_f = idx
            if v_rh > best_rh:
                best_rh = v_rh
                best_rh_f = idx
            if v_lh > best_lh:
                best_lh = v_lh
                best_lh_f = idx
            if v_rt > best_rt:
                best_rt = v_rt
                best_rt_f = idx

        score0[i] = best_lt + best_rh
        score1[i] = best_lh + best_rt
        fl0[i] = best_lt_f
        fr0[i] = best_rh_f
        fl1[i] = best_lh_f
        fr1[i] = best_rt_f

    return _solve_alternating_reference(score0, score1, fl0, fr0, fl1, fr1)


class TestNativeFormulaValidation(unittest.TestCase):
    def test_solve_alternating_path_reference_parity(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            n = int(rng.integers(1, 50))
            score0 = rng.normal(size=n).astype(float).tolist()
            score1 = rng.normal(size=n).astype(float).tolist()
            fl0 = rng.integers(0, 400, size=n).astype(int).tolist()
            fr0 = rng.integers(0, 400, size=n).astype(int).tolist()
            fl1 = rng.integers(0, 400, size=n).astype(int).tolist()
            fr1 = rng.integers(0, 400, size=n).astype(int).tolist()

            n_states, n_lf, n_rf, n_total = _gait_native.solve_alternating_path(
                score0, score1, fl0, fr0, fl1, fr1
            )
            p_states, p_lf, p_rf, p_total = _solve_alternating_reference(
                score0, score1, fl0, fr0, fl1, fr1
            )

            self.assertEqual(list(n_states), p_states)
            self.assertEqual(list(n_lf), p_lf)
            self.assertEqual(list(n_rf), p_rf)
            self.assertAlmostEqual(float(n_total), float(p_total), places=10)

    def test_solve_prob_intervals_reference_parity(self):
        rng = np.random.default_rng(43)
        for _ in range(50):
            n_frames = int(rng.integers(60, 220))
            n_intervals = int(rng.integers(3, 18))
            starts = np.sort(rng.integers(0, n_frames - 2, size=n_intervals).astype(np.int32))
            ends = starts + rng.integers(1, 25, size=n_intervals).astype(np.int32)
            ends = np.minimum(ends, n_frames).astype(np.int32)

            l_hs = rng.uniform(1e-5, 1.0, size=n_frames).astype(np.float64)
            l_to = rng.uniform(1e-5, 1.0, size=n_frames).astype(np.float64)
            r_hs = rng.uniform(1e-5, 1.0, size=n_frames).astype(np.float64)
            r_to = rng.uniform(1e-5, 1.0, size=n_frames).astype(np.float64)

            n_states, n_lf, n_rf, n_total = _gait_native.solve_prob_intervals(
                starts, ends, l_hs, l_to, r_hs, r_to
            )
            p_states, p_lf, p_rf, p_total = _solve_prob_intervals_reference(
                starts.tolist(),
                ends.tolist(),
                l_hs,
                l_to,
                r_hs,
                r_to,
            )

            self.assertEqual(list(n_states), p_states)
            self.assertEqual(list(n_lf), p_lf)
            self.assertEqual(list(n_rf), p_rf)
            self.assertAlmostEqual(float(n_total), float(p_total), places=10)

    def test_solve_bis_intervals_reference_parity(self):
        rng = np.random.default_rng(44)
        hs_mu, hs_sigma, hs_weight = 0.20, 0.18, 1.5
        to_mu, to_sigma, to_weight = 0.85, 0.20, 1.5

        for _ in range(50):
            n_frames = int(rng.integers(60, 220))
            n_intervals = int(rng.integers(3, 18))
            starts = np.sort(rng.integers(0, n_frames - 2, size=n_intervals).astype(np.int32))
            ends = starts + rng.integers(1, 25, size=n_intervals).astype(np.int32)
            ends = np.minimum(ends, n_frames).astype(np.int32)

            l_hs = rng.normal(size=n_frames).astype(np.float64)
            l_to = rng.normal(size=n_frames).astype(np.float64)
            r_hs = rng.normal(size=n_frames).astype(np.float64)
            r_to = rng.normal(size=n_frames).astype(np.float64)

            n_states, n_lf, n_rf, n_total = _gait_native.solve_bis_intervals(
                starts,
                ends,
                l_hs,
                l_to,
                r_hs,
                r_to,
                hs_mu,
                hs_sigma,
                hs_weight,
                to_mu,
                to_sigma,
                to_weight,
            )
            p_states, p_lf, p_rf, p_total = _solve_bis_intervals_reference(
                starts.tolist(),
                ends.tolist(),
                l_hs,
                l_to,
                r_hs,
                r_to,
                hs_mu,
                hs_sigma,
                hs_weight,
                to_mu,
                to_sigma,
                to_weight,
            )

            self.assertEqual(list(n_states), p_states)
            self.assertEqual(list(n_lf), p_lf)
            self.assertEqual(list(n_rf), p_rf)
            self.assertAlmostEqual(float(n_total), float(p_total), places=10)

    def test_find_peaks_filtered_known_distance_case(self):
        x = np.zeros(70, dtype=np.float64)
        x[17] = 10.0
        x[32] = 9.0
        x[46] = 9.5

        sci, _ = find_peaks(x, distance=20, prominence=0.01)
        nat = np.asarray(
            _gait_native.find_peaks_filtered(x, 20, 0.01, 1),
            dtype=np.int32,
        )
        self.assertEqual(sci.tolist(), [17, 46])
        self.assertEqual(nat.tolist(), sci.tolist())


if __name__ == "__main__":
    unittest.main()

