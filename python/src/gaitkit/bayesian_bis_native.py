"""
Native-accelerated Bayesian BIS detector (non-intrusive integration).

This module does not modify original detector code. It subclasses the
reference Python implementation and overrides only the two Viterbi-like
assignment steps with an optimized alternating-state solver.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

try:
    from gaitkit.native import _gait_native as _native_solver
    _HAS_NATIVE = True
except Exception:
    _native_solver = None
    _HAS_NATIVE = False

try:
    from gaitkit.detectors.bayesian_bis import (
        BayesianBisGaitDetector,
        GaitEvent,
        HS_PHASE_PRIOR,
        TO_PHASE_PRIOR,
    )
except Exception:
    from gaitkit.detectors.bayesian_bis import (
        BayesianBisGaitDetector,
        GaitEvent,
        HS_PHASE_PRIOR,
        TO_PHASE_PRIOR,
    )


_STATE_LABELS = [("TO", "HS"), ("HS", "TO")]
_INF = -1e9


class BayesianBisNativeGaitDetector(BayesianBisGaitDetector):
    """Drop-in replacement for BayesianBisGaitDetector with faster assignment."""

    def _extract_and_smooth(self, af):
        n = len(af)
        ap = self._ap_axis

        lk = np.empty(n, dtype=np.float64)
        rk = np.empty(n, dtype=np.float64)
        la = np.empty(n, dtype=np.float64)
        ra = np.empty(n, dtype=np.float64)

        for i, f in enumerate(af):
            lp = f.landmark_positions
            if lp:
                lk[i] = lp["left_knee"][ap]
                rk[i] = lp["right_knee"][ap]
                la[i] = lp["left_ankle"][ap]
                ra[i] = lp["right_ankle"][ap]
            else:
                lk[i] = 0.5
                rk[i] = 0.5
                la[i] = 0.5
                ra[i] = 0.5

        if n > self.smoothing_window:
            return {
                "left_knee": savgol_filter(lk, self.smoothing_window, 3),
                "right_knee": savgol_filter(rk, self.smoothing_window, 3),
                "left_ankle": savgol_filter(la, self.smoothing_window, 3),
                "right_ankle": savgol_filter(ra, self.smoothing_window, 3),
            }
        s = self.smoothing_window / 6.0
        return {
            "left_knee": gaussian_filter1d(lk, s),
            "right_knee": gaussian_filter1d(rk, s),
            "left_ankle": gaussian_filter1d(la, s),
            "right_ankle": gaussian_filter1d(ra, s),
        }

    def _extract_biomechanical_features(self, af):
        n = len(af)
        ap = self._ap_axis
        vert = self._vert_axis
        direction = self._direction

        px = np.zeros(n, dtype=np.float64)

        left_knee_ap = np.zeros(n, dtype=np.float64)
        right_knee_ap = np.zeros(n, dtype=np.float64)
        left_heel_ap = np.zeros(n, dtype=np.float64)
        right_heel_ap = np.zeros(n, dtype=np.float64)
        left_ankle_ap = np.zeros(n, dtype=np.float64)
        right_ankle_ap = np.zeros(n, dtype=np.float64)
        left_heel_z = np.zeros(n, dtype=np.float64)
        right_heel_z = np.zeros(n, dtype=np.float64)
        left_toe_z = np.zeros(n, dtype=np.float64)
        right_toe_z = np.zeros(n, dtype=np.float64)
        left_ankle_z = np.zeros(n, dtype=np.float64)
        right_ankle_z = np.zeros(n, dtype=np.float64)
        left_knee_angle = np.empty(n, dtype=np.float64)
        right_knee_angle = np.empty(n, dtype=np.float64)
        left_ankle_angle = np.empty(n, dtype=np.float64)
        right_ankle_angle = np.empty(n, dtype=np.float64)

        has_heel_left = False
        has_heel_right = False
        has_toe_left = False
        has_toe_right = False
        has_ankle_left = False
        has_ankle_right = False

        for i, f in enumerate(af):
            left_knee_angle[i] = f.left_knee_angle
            right_knee_angle[i] = f.right_knee_angle
            left_ankle_angle[i] = f.left_ankle_angle
            right_ankle_angle[i] = f.right_ankle_angle

            lp = f.landmark_positions
            if not lp:
                continue

            lh = lp.get("left_hip", (0, 0, 0))
            rh = lp.get("right_hip", (0, 0, 0))
            px[i] = (lh[ap] + rh[ap]) / 2.0

            lk = lp.get("left_knee", None)
            rk = lp.get("right_knee", None)
            lh_m = lp.get("left_heel", None)
            rh_m = lp.get("right_heel", None)
            la_m = lp.get("left_ankle", None)
            ra_m = lp.get("right_ankle", None)
            lt = lp.get("left_toe", None)
            rt = lp.get("right_toe", None)

            if lk:
                left_knee_ap[i] = lk[ap]
            if rk:
                right_knee_ap[i] = rk[ap]
            if lh_m:
                left_heel_ap[i] = lh_m[ap]
                left_heel_z[i] = lh_m[vert]
                has_heel_left = True
            if rh_m:
                right_heel_ap[i] = rh_m[ap]
                right_heel_z[i] = rh_m[vert]
                has_heel_right = True
            if la_m:
                left_ankle_ap[i] = la_m[ap]
                left_ankle_z[i] = la_m[vert]
                has_ankle_left = True
            if ra_m:
                right_ankle_ap[i] = ra_m[ap]
                right_ankle_z[i] = ra_m[vert]
                has_ankle_right = True
            if lt:
                left_toe_z[i] = lt[vert]
                has_toe_left = True
            if rt:
                right_toe_z[i] = rt[vert]
                has_toe_right = True

        thigh_lengths = []
        for f in af[:: max(1, n // 20)]:
            lp = f.landmark_positions
            if lp:
                for side in ["left", "right"]:
                    hip = np.array(lp.get(f"{side}_hip", (0, 0, 0)))
                    knee = np.array(lp.get(f"{side}_knee", (0, 0, 0)))
                    d = np.linalg.norm(hip - knee)
                    if d > 50:
                        thigh_lengths.append(d)
        thigh_len = np.mean(thigh_lengths) if thigh_lengths else 400.0

        sigma_s = max(1.0, self.smoothing_window / 6.0)
        features = {}
        for side in ["left", "right"]:
            if side == "left":
                knee_ap = left_knee_ap
                heel_ap = left_heel_ap
                ankle_ap = left_ankle_ap
                heel_z = left_heel_z
                toe_z = left_toe_z
                ankle_z = left_ankle_z
                knee_angle = left_knee_angle
                ankle_angle = left_ankle_angle
                has_heel = has_heel_left
                has_toe = has_toe_left
                has_ankle = has_ankle_left
            else:
                knee_ap = right_knee_ap
                heel_ap = right_heel_ap
                ankle_ap = right_ankle_ap
                heel_z = right_heel_z
                toe_z = right_toe_z
                ankle_z = right_ankle_z
                knee_angle = right_knee_angle
                ankle_angle = right_ankle_angle
                has_heel = has_heel_right
                has_toe = has_toe_right
                has_ankle = has_ankle_right

            if has_heel:
                knee_heel_ap_raw = (heel_ap - knee_ap) * direction
            else:
                knee_heel_ap_raw = (ankle_ap - knee_ap) * direction
            knee_heel_ap = knee_heel_ap_raw / thigh_len
            knee_heel_ap = gaussian_filter1d(knee_heel_ap, sigma_s)

            ankle_rel = (ankle_ap - px) * direction
            ankle_rel_smooth = gaussian_filter1d(ankle_rel, sigma_s)
            ankle_ap_vel = np.gradient(ankle_rel_smooth) * self.fps / thigh_len
            ankle_ap_vel = gaussian_filter1d(ankle_ap_vel, sigma_s * 0.5)

            if has_heel:
                heel_z_smooth = gaussian_filter1d(heel_z, sigma_s)
                hz_min, hz_max = np.min(heel_z_smooth), np.max(heel_z_smooth)
                hz_range = hz_max - hz_min if hz_max > hz_min else 1.0
                heel_vert_norm = (heel_z_smooth - hz_min) / hz_range
                heel_vert_vel = np.gradient(heel_vert_norm) * self.fps
            else:
                heel_vert_norm = np.zeros(n, dtype=np.float64)
                heel_vert_vel = np.zeros(n, dtype=np.float64)

            toe_z_smooth = None
            if has_toe:
                toe_z_smooth = gaussian_filter1d(toe_z, sigma_s)
                tz_min, tz_max = np.min(toe_z_smooth), np.max(toe_z_smooth)
                tz_range = tz_max - tz_min if tz_max > tz_min else 1.0
                toe_vert_norm = (toe_z_smooth - tz_min) / tz_range
                toe_vert_vel = np.gradient(toe_vert_norm) * self.fps
            else:
                toe_vert_norm = np.zeros(n, dtype=np.float64)
                toe_vert_vel = np.zeros(n, dtype=np.float64)

            knee_angle_smooth = gaussian_filter1d(knee_angle, sigma_s)
            knee_angle_vel = np.gradient(knee_angle_smooth) * self.fps

            omega_knee = knee_angle_vel
            ankle_angle_smooth = gaussian_filter1d(ankle_angle, sigma_s)
            omega_ankle = np.gradient(ankle_angle_smooth) * self.fps

            if has_toe:
                v_toe_vert = np.gradient(toe_z_smooth) * self.fps
            else:
                v_toe_vert = np.zeros(n, dtype=np.float64)

            if has_ankle:
                ankle_z_smooth = gaussian_filter1d(ankle_z, sigma_s)
                v_ankle_vert = np.gradient(ankle_z_smooth) * self.fps
            else:
                v_ankle_vert = np.zeros(n, dtype=np.float64)

            abs_omega_knee = np.abs(omega_knee)
            abs_v_toe = np.abs(v_toe_vert)
            abs_v_ankle = np.abs(v_ankle_vert)
            denom = abs_omega_knee + abs_v_toe + abs_v_ankle + 1e-10

            R_toe = abs_v_toe / denom
            R_knee = abs_omega_knee / denom
            delta_omega = omega_knee - omega_ankle

            vel_magnitude = abs_omega_knee + abs_v_toe + abs_v_ankle
            vel_med = np.median(vel_magnitude[vel_magnitude > 0]) if np.any(vel_magnitude > 0) else 1.0
            vel_threshold = vel_med * 0.25
            vel_confidence = np.clip(vel_magnitude / (vel_threshold + 1e-10), 0.0, 1.0)

            features[side] = {
                "knee_heel_ap": knee_heel_ap,
                "ankle_ap_vel": ankle_ap_vel,
                "heel_vert": heel_vert_norm,
                "heel_vert_vel": heel_vert_vel,
                "toe_vert": toe_vert_norm,
                "toe_vert_vel": toe_vert_vel,
                "knee_angle": knee_angle_smooth,
                "knee_angle_vel": knee_angle_vel,
                "has_heel": has_heel,
                "has_toe": has_toe,
                "R_toe": R_toe,
                "R_knee": R_knee,
                "delta_omega": delta_omega,
                "vel_confidence": vel_confidence,
                "has_ankle": has_ankle,
            }
        return features

    def _event_probs_zeni(self, af):
        n = len(af)
        ap = getattr(self, "_ap_axis", 0)
        direction = getattr(self, "_direction", 1)

        lar = np.empty(n, dtype=np.float64)
        rar = np.empty(n, dtype=np.float64)

        for i, f in enumerate(af):
            lp = f.landmark_positions
            lh = lp.get("left_hip", (0.5, 0, 0))
            rh = lp.get("right_hip", (0.5, 0, 0))
            px = (lh[ap] + rh[ap]) / 2.0
            lar[i] = lp.get("left_ankle", (0.5, 0, 0))[ap] - px
            rar[i] = lp.get("right_ankle", (0.5, 0, 0))[ap] - px

        if direction < 0:
            lar = -lar
            rar = -rar
        if n > self.smoothing_window:
            sigma = self.smoothing_window / 4
            lar = gaussian_filter1d(lar, sigma)
            rar = gaussian_filter1d(rar, sigma)

        lv = np.gradient(lar)
        rv = np.gradient(rar)

        probs = {
            "left_hs": np.zeros(n, dtype=np.float64),
            "left_to": np.zeros(n, dtype=np.float64),
            "right_hs": np.zeros(n, dtype=np.float64),
            "right_to": np.zeros(n, dtype=np.float64),
        }

        for side, ar, v in [("left", lar, lv), ("right", rar, rv)]:
            mn, mx = np.min(ar), np.max(ar)
            r = mx - mn if mx > mn else 1
            pn = (ar - mn) / r
            va = np.abs(v)
            vm = np.max(va)
            if vm <= 0:
                vm = 1
            vn = 1 - va / vm
            probs[side + "_hs"] = pn ** 2 * vn ** 0.5
            probs[side + "_to"] = (1 - pn) ** 2 * vn ** 0.5

        for k in probs:
            pm = np.max(probs[k])
            if pm > 0:
                probs[k] = probs[k] / pm * 0.99
            probs[k] = np.clip(probs[k], 0.01, 0.99)
        return probs

    def _calc_p_signal(self, traj):
        la = (traj["left_knee"] + traj["left_ankle"]) / 2
        ra = (traj["right_knee"] + traj["right_ankle"]) / 2
        diff = la - ra
        self.debug_data["diff"] = diff

        dv = np.gradient(diff)
        if len(diff) > 5:
            dv = savgol_filter(dv, 5, 2)

        if _HAS_NATIVE:
            ps_raw = np.asarray(
                _native_solver.calc_p_signal_raw(
                    np.asarray(diff, dtype=np.float64),
                    np.asarray(dv, dtype=np.float64),
                    int(self.min_crossing_frames),
                    float(self.MAD_TO_SIGMA_FACTOR),
                    float(self.LOCAL_SIGMA_BLEND_LOCAL),
                    float(self.LOCAL_SIGMA_BLEND_GLOBAL),
                    float(self.LOCAL_SIGMA_MIN_RATIO),
                    float(self.P_SIGNAL_PROXIMITY_BANDWIDTH),
                ),
                dtype=np.float64,
            )
            ps = gaussian_filter1d(ps_raw, sigma=self.P_SIGNAL_SMOOTHING_SIGMA)
            self.debug_data["p_signal"] = ps.copy()
            return ps

        return super()._calc_p_signal(traj)

    def _solve_alternating_path(
        self,
        score0: Sequence[float],
        score1: Sequence[float],
        fl0: Sequence[int],
        fr0: Sequence[int],
        fl1: Sequence[int],
        fr1: Sequence[int],
    ) -> Tuple[List[int], List[int], List[int], float]:
        if _HAS_NATIVE:
            states, left_frames, right_frames, total = _native_solver.solve_alternating_path(
                list(score0), list(score1), list(fl0), list(fr0), list(fl1), list(fr1)
            )
            return list(states), list(left_frames), list(right_frames), float(total)

        # Python fallback with the exact same constrained model:
        # states must strictly alternate between 0 and 1.
        n = len(score0)
        if n == 0:
            return [], [], [], _INF
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
        return states, lf, rf, float(total0 if start == 0 else total1)

    def _bayesian_event_assignment(self, crossings, features, af):
        n_intervals = len(crossings)
        if n_intervals == 0:
            return []

        if _HAS_NATIVE:
            n = len(af)
            left = features["left"]
            right = features["right"]
            ll_l_hs = self._compute_log_likelihood_hs_array(left, 0, n)
            ll_l_to = self._compute_log_likelihood_to_array(left, 0, n)
            ll_r_hs = self._compute_log_likelihood_hs_array(right, 0, n)
            ll_r_to = self._compute_log_likelihood_to_array(right, 0, n)
            starts = np.fromiter((iv.start_frame for iv in crossings), dtype=np.int32, count=n_intervals)
            ends = np.fromiter((iv.end_frame for iv in crossings), dtype=np.int32, count=n_intervals)
            states, left_frames, right_frames, _ = _native_solver.solve_bis_intervals(
                starts,
                ends,
                ll_l_hs,
                ll_l_to,
                ll_r_hs,
                ll_r_to,
                float(HS_PHASE_PRIOR.mu),
                float(HS_PHASE_PRIOR.sigma),
                float(HS_PHASE_PRIOR.weight),
                float(TO_PHASE_PRIOR.mu),
                float(TO_PHASE_PRIOR.sigma),
                float(TO_PHASE_PRIOR.weight),
            )
            events = []
            for i in range(n_intervals - 1, -1, -1):
                st = int(states[i])
                sl, sr = _STATE_LABELS[st]
                fl = int(left_frames[i])
                fr = int(right_frames[i])
                etl = "heel_strike" if sl == "HS" else "toe_off"
                etr = "heel_strike" if sr == "HS" else "toe_off"
                events.append(GaitEvent(fl, fl / self.fps, etl, "left", 0.8))
                events.append(GaitEvent(fr, fr / self.fps, etr, "right", 0.8))
            events.sort(key=lambda e: e.frame_index)
            return events

        score0, score1 = [], []
        fl0, fr0 = [], []
        fl1, fr1 = [], []
        for iv in crossings:
            s, e = iv.start_frame, iv.end_frame
            sc0, l0, r0 = self._best_bayesian_frames(s, e, "TO", "HS", features)
            sc1, l1, r1 = self._best_bayesian_frames(s, e, "HS", "TO", features)
            score0.append(sc0)
            score1.append(sc1)
            fl0.append(l0)
            fr0.append(r0)
            fl1.append(l1)
            fr1.append(r1)

        states, left_frames, right_frames, _ = self._solve_alternating_path(
            score0, score1, fl0, fr0, fl1, fr1
        )
        if not states:
            return []

        events = []
        for i in range(n_intervals - 1, -1, -1):
            st = states[i]
            sl, sr = _STATE_LABELS[st]
            fl = int(left_frames[i])
            fr = int(right_frames[i])
            etl = "heel_strike" if sl == "HS" else "toe_off"
            etr = "heel_strike" if sr == "HS" else "toe_off"
            events.append(GaitEvent(fl, fl / self.fps, etl, "left", 0.8))
            events.append(GaitEvent(fr, fr / self.fps, etr, "right", 0.8))
        events.sort(key=lambda e: e.frame_index)
        return events

    def _viterbi_fallback(self, crossings, ep):
        ni = len(crossings)
        if ni == 0:
            return []

        if _HAS_NATIVE:
            starts = np.fromiter((iv.start_frame for iv in crossings), dtype=np.int32, count=ni)
            ends = np.fromiter((iv.end_frame for iv in crossings), dtype=np.int32, count=ni)
            states, left_frames, right_frames, _ = _native_solver.solve_prob_intervals(
                starts,
                ends,
                ep["left_hs"],
                ep["left_to"],
                ep["right_hs"],
                ep["right_to"],
            )
            events = []
            for i in range(ni - 1, -1, -1):
                st = int(states[i])
                sl, sr = _STATE_LABELS[st]
                fl = int(left_frames[i])
                fr = int(right_frames[i])
                etl = "heel_strike" if sl == "HS" else "toe_off"
                etr = "heel_strike" if sr == "HS" else "toe_off"
                p_l = float(ep[f"left_{sl.lower()}"][fl])
                p_r = float(ep[f"right_{sr.lower()}"][fr])
                events.append(GaitEvent(fl, fl / self.fps, etl, "left", p_l))
                events.append(GaitEvent(fr, fr / self.fps, etr, "right", p_r))
            events.sort(key=lambda e: e.frame_index)
            return events

        score0, score1 = [], []
        fl0, fr0 = [], []
        fl1, fr1 = [], []
        for iv in crossings:
            s, e = iv.start_frame, iv.end_frame
            sc0, l0, r0 = self._best_frames_ep(s, e, "TO", "HS", ep)
            sc1, l1, r1 = self._best_frames_ep(s, e, "HS", "TO", ep)
            score0.append(sc0)
            score1.append(sc1)
            fl0.append(l0)
            fr0.append(r0)
            fl1.append(l1)
            fr1.append(r1)

        states, left_frames, right_frames, _ = self._solve_alternating_path(
            score0, score1, fl0, fr0, fl1, fr1
        )
        if not states:
            return []

        events = []
        for i in range(ni - 1, -1, -1):
            st = states[i]
            sl, sr = _STATE_LABELS[st]
            fl = int(left_frames[i])
            fr = int(right_frames[i])
            etl = "heel_strike" if sl == "HS" else "toe_off"
            etr = "heel_strike" if sr == "HS" else "toe_off"
            p_l = float(ep[f"left_{sl.lower()}"][fl])
            p_r = float(ep[f"right_{sr.lower()}"][fr])
            events.append(GaitEvent(fl, fl / self.fps, etl, "left", p_l))
            events.append(GaitEvent(fr, fr / self.fps, etr, "right", p_r))
        events.sort(key=lambda e: e.frame_index)
        return events
