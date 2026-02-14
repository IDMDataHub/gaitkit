# -*- coding: utf-8 -*-
"""
Bayesian Gait Event Detector -- BIS V2 (Bayesian with Improved Signals V2).

Extends BIS with 4 ratio-based velocity features in Stage 2:
  1. R_toe  = |v_toe_vert| / (|omega_knee| + |v_toe_vert| + |v_ankle_vert|)
  2. R_knee = |omega_knee| / (|omega_knee| + |v_toe_vert| + |v_ankle_vert|)
  3. delta_omega = omega_knee - omega_ankle  (angular velocity difference)
  4. knee_vel_norm = |omega_knee| / local_max(|omega_knee|)  (velocity intensity [0,1])
     -> At TO, knee_vel_norm ~ 0.7 (knee moving fast relative to local max)
     -> At HS, knee_vel_norm ~ 0.25 (knee nearly stopped)
     -> Cohen's d = 1.69 on PD, 93% consistency for TO localization

Author: Frederic Fer (f.fer@institut-myologie.org)
Affiliation: Myodata, Institut de Myologie, Paris, France
License: Apache-2.0
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter1d
import logging

from .axis_utils import detect_axes, detect_walking_direction

logger = logging.getLogger(__name__)


# ===========================================================================
# Data classes (same as V3.6 for compatibility)
# ===========================================================================

@dataclass
class GaitEvent:
    frame_index: int
    time: float
    event_type: str  # 'heel_strike' or 'toe_off'
    side: str  # 'left' or 'right'
    probability: float


@dataclass
class GaitCycle:
    cycle_id: int
    side: str
    start_frame: int
    toe_off_frame: Optional[int]
    end_frame: int
    start_time: float
    toe_off_time: Optional[float]
    end_time: float
    duration: float
    stance_duration: Optional[float]
    swing_duration: Optional[float]
    stance_percentage: Optional[float]
    swing_percentage: Optional[float]


@dataclass
class CrossingInterval:
    start_frame: int
    end_frame: int
    confidence: float


@dataclass
class RhythmModel:
    model_type: str
    T1: float
    T2: Optional[float] = None
    likelihood: float = 0.0
    sigma: float = 0.0
    first_crossing: int = 0


# ===========================================================================
# Biomechanical Prior Definitions
# ===========================================================================

@dataclass
class GaussianPrior:
    mu: float
    sigma: float
    weight: float = 1.0

    def log_prob(self, x: float) -> float:
        z = (x - self.mu) / max(self.sigma, 1e-10)
        return self.weight * (-0.5 * z * z)

    def log_prob_array(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mu) / max(self.sigma, 1e-10)
        return self.weight * (-0.5 * z * z)


# Hard-coded biomechanical priors (V1 originals)
HS_PRIORS = {
    'knee_heel_ap': GaussianPrior(mu=0.12, sigma=0.15, weight=2.0),
    'ankle_ap_vel': GaussianPrior(mu=-0.8, sigma=0.5, weight=1.5),
    'heel_vert': GaussianPrior(mu=0.02, sigma=0.1, weight=1.5),
    'heel_vert_vel': GaussianPrior(mu=-1.0, sigma=2.0, weight=0.5),
    'knee_angle': GaussianPrior(mu=5.0, sigma=8.0, weight=1.0),
    'knee_angle_vel': GaussianPrior(mu=0.0, sigma=80.0, weight=0.3),
    # --- NEW ratio-based features ---
    # Wide sigmas + low weights for robustness to pathological gait
    'R_toe': GaussianPrior(mu=0.61, sigma=0.25, weight=0.6),
    'R_knee': GaussianPrior(mu=0.11, sigma=0.20, weight=0.6),
    'delta_omega': GaussianPrior(mu=0.0, sigma=80.0, weight=0.25),
    # --- V2: normalized knee angular velocity (z-scored) ---
    # At HS, knee is near full extension, angular velocity near zero
    # z-score mean ~ -0.28, conservative weight to avoid regression
    'knee_vel_norm': GaussianPrior(mu=0.25, sigma=0.35, weight=0.10),
}

TO_PRIORS = {
    'knee_heel_ap': GaussianPrior(mu=-0.72, sigma=0.15, weight=2.0),
    'ankle_ap_vel': GaussianPrior(mu=0.8, sigma=0.6, weight=1.5),
    'toe_vert': GaussianPrior(mu=0.3, sigma=0.25, weight=0.8),
    'toe_vert_vel': GaussianPrior(mu=3.5, sigma=2.0, weight=2.5),
    'knee_angle': GaussianPrior(mu=43.0, sigma=10.0, weight=1.0),
    'knee_angle_vel': GaussianPrior(mu=200.0, sigma=150.0, weight=1.0),
    # --- NEW ratio-based features ---
    'R_toe': GaussianPrior(mu=0.27, sigma=0.25, weight=0.6),
    'R_knee': GaussianPrior(mu=0.30, sigma=0.20, weight=0.6),
    'delta_omega': GaussianPrior(mu=100.0, sigma=120.0, weight=0.25),
    # --- V2: normalized knee angular velocity (z-scored) ---
    # At TO, knee is accelerating into flexion, high positive velocity
    # z-score mean ~ +0.93, moderate weight for TO discrimination
    'knee_vel_norm': GaussianPrior(mu=0.70, sigma=0.30, weight=0.10),
}

HS_PHASE_PRIOR = GaussianPrior(mu=0.20, sigma=0.18, weight=1.5)
TO_PHASE_PRIOR = GaussianPrior(mu=0.85, sigma=0.20, weight=1.5)


# ===========================================================================
# Main detector class
# ===========================================================================

class BayesianBisV2GaitDetector:
    PEAK_HEIGHT_STRICT = 0.1
    PEAK_PROMINENCE_STRICT = 0.05
    PEAK_HEIGHT_RELAXED = 0.05
    PEAK_PROMINENCE_RELAXED = 0.02
    PEAK_HEIGHT_SEGMENT = 0.1
    PEAK_PROMINENCE_SEGMENT = 0.03
    SEGMENT_BREAK_RATIO = 2.0
    REINFORCEMENT_DECAY = 0.5
    REINFORCEMENT_MAX_HOPS = 5
    REINFORCEMENT_MAX_CONTRIBUTION = 0.1
    MAD_TO_SIGMA_FACTOR = 1.4826
    LOCAL_SIGMA_BLEND_LOCAL = 0.7
    LOCAL_SIGMA_BLEND_GLOBAL = 0.3
    LOCAL_SIGMA_MIN_RATIO = 0.01
    P_SIGNAL_PROXIMITY_BANDWIDTH = 0.3
    P_SIGNAL_SMOOTHING_SIGMA = 2.0
    BOUNDARY_PEAK_HEIGHT_LEADING = 0.40
    BOUNDARY_PEAK_PROMINENCE_LEADING = 0.10
    BOUNDARY_PEAK_HEIGHT_TRAILING = 0.35
    BOUNDARY_PEAK_PROMINENCE_TRAILING = 0.05
    BOUNDARY_FALLBACK_LEADING_THRESHOLD = 0.7
    BOUNDARY_FALLBACK_TRAILING_THRESHOLD = 0.70
    BOUNDARY_MIN_LEADING_FRAME = 0
    GAP_THRESHOLD_RATIO = 1.8
    GAP_PEAK_HEIGHT = 0.5
    GAP_PEAK_PROMINENCE = 0.15
    TRAILING_TO_SUPPRESS_RATIO = 0.25
    VERTICAL_CORRECTION_SHRINKAGE = 0.5
    VERTICAL_CORRECTION_MAX_RATIO = 0.08
    VERTICAL_CORRECTION_MIN_SHIFT = 2
    VERTICAL_RANGE_THRESHOLD = 5.0

    def __init__(self, fps: float = 100.0, smoothing_window: int = 11,
                 min_crossing_distance: float = 0.2,
                 rhythm_sigma_ratio: float = 0.15) -> None:
        self.fps = fps
        self.smoothing_window = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1
        self.min_crossing_frames = int(min_crossing_distance * fps)
        self.rhythm_sigma_ratio = rhythm_sigma_ratio
        self.debug_data: Dict = {}

    def detect_gait_events(self, angle_frames):
        n = len(angle_frames)
        if n < 30:
            return [], [], []

        has_landmarks = (angle_frames[0].landmark_positions is not None and
                         "left_ankle" in angle_frames[0].landmark_positions)
        if has_landmarks:
            self._ap_axis, self._vert_axis = detect_axes(angle_frames)
            self._direction = detect_walking_direction(angle_frames, self._ap_axis, self.fps)
        else:
            self._ap_axis = 0
            self._vert_axis = 2
            self._direction = 1

        traj = self._extract_and_smooth(angle_frames)
        crossings, rm = self._detect_crossings_with_fallback(traj)

        if len(crossings) < 2:
            return [], [], []

        if has_landmarks:
            features = self._extract_biomechanical_features(angle_frames)
            events = self._bayesian_event_assignment(crossings, features, angle_frames)
            # Use Zeni-style probs for boundary/gap (Bayesian probs saturate)
            ep = self._event_probs_zeni(angle_frames)
            ep = self._reinforce_event_probs(ep, 2, 0.1)
        else:
            ep = self._event_probs_angles(angle_frames)
            events = self._viterbi_fallback(crossings, ep)

        be = self._boundary_events(crossings, events, ep, n, rm)
        # Refine boundary events using Bayesian features
        if has_landmarks and be:
            be = self._refine_boundary_bayesian(be, features, crossings, rm)
        events.extend(be)
        events.sort(key=lambda e: e.frame_index)

        ge = self._gap_events(events, ep, n, rm)
        events.extend(ge)
        events.sort(key=lambda e: e.frame_index)

        hs = [e for e in events if e.event_type == "heel_strike"]
        to = [e for e in events if e.event_type == "toe_off"]

        return hs, to, self._build_cycles(hs, to)

    def _extract_and_smooth(self, af):
        n = len(af)
        ap = self._ap_axis
        lk = np.array([f.landmark_positions["left_knee"][ap] if f.landmark_positions else 0.5 for f in af])
        rk = np.array([f.landmark_positions["right_knee"][ap] if f.landmark_positions else 0.5 for f in af])
        la = np.array([f.landmark_positions["left_ankle"][ap] if f.landmark_positions else 0.5 for f in af])
        ra = np.array([f.landmark_positions["right_ankle"][ap] if f.landmark_positions else 0.5 for f in af])
        if n > self.smoothing_window:
            return {"left_knee": savgol_filter(lk, self.smoothing_window, 3),
                    "right_knee": savgol_filter(rk, self.smoothing_window, 3),
                    "left_ankle": savgol_filter(la, self.smoothing_window, 3),
                    "right_ankle": savgol_filter(ra, self.smoothing_window, 3)}
        s = self.smoothing_window / 6.0
        return {"left_knee": gaussian_filter1d(lk, s), "right_knee": gaussian_filter1d(rk, s),
                "left_ankle": gaussian_filter1d(la, s), "right_ankle": gaussian_filter1d(ra, s)}

    def _calc_p_signal(self, traj):
        la = (traj["left_knee"] + traj["left_ankle"]) / 2
        ra = (traj["right_knee"] + traj["right_ankle"]) / 2
        diff = la - ra
        self.debug_data["diff"] = diff
        dv = np.gradient(diff)
        if len(diff) > 5:
            dv = savgol_filter(dv, 5, 2)
        ds_global = np.std(diff)
        n = len(diff)
        if n > 20 and ds_global > 0:
            half_win = max(self.min_crossing_frames * 2, 30)
            sigma_local = np.full(n, ds_global)
            for i in range(n):
                ws = max(0, i - half_win)
                we = min(n, i + half_win + 1)
                local_seg = diff[ws:we]
                if len(local_seg) > 10:
                    med = np.median(local_seg)
                    mad = np.median(np.abs(local_seg - med))
                    sigma_mad = self.MAD_TO_SIGMA_FACTOR * mad
                    if sigma_mad > self.LOCAL_SIGMA_MIN_RATIO * ds_global:
                        sigma_local[i] = (self.LOCAL_SIGMA_BLEND_LOCAL * sigma_mad
                                          + self.LOCAL_SIGMA_BLEND_GLOBAL * ds_global)
            sigma_kernel = sigma_local * self.P_SIGNAL_PROXIMITY_BANDWIDTH
            sigma_kernel = np.maximum(sigma_kernel, 0.01)
            pp = np.exp(-0.5 * (diff / sigma_kernel)**2)
            va = np.abs(dv)
            vel_half_win = max(self.min_crossing_frames, 15)
            va_local_max = np.ones(n)
            for i in range(n):
                ws = max(0, i - vel_half_win)
                we = min(n, i + vel_half_win + 1)
                local_max = np.max(va[ws:we])
                va_local_max[i] = max(local_max, 1e-10)
            ps = pp * (va / va_local_max)
        else:
            pp = np.exp(-0.5 * (diff / (ds_global * self.P_SIGNAL_PROXIMITY_BANDWIDTH))**2) if ds_global > 0 else np.ones_like(diff)
            va = np.abs(dv)
            vm = np.max(va) if np.max(va) > 0 else 1
            ps = pp * (va / vm)
        pm = np.max(ps)
        if pm > 0:
            ps = ps / pm
        ps = gaussian_filter1d(ps, sigma=self.P_SIGNAL_SMOOTHING_SIGMA)
        self.debug_data["p_signal"] = ps.copy()
        return ps

    def _detect_crossings_with_fallback(self, traj):
        n = len(traj["left_knee"])
        ps = self._calc_p_signal(traj)
        pc = ps.copy()
        pks, _ = find_peaks(pc, height=self.PEAK_HEIGHT_STRICT,
                            distance=self.min_crossing_frames,
                            prominence=self.PEAK_PROMINENCE_STRICT)
        cands = list(pks)
        if len(cands) >= 2:
            return self._process_crossings(cands, pc, traj, n)
        pks, _ = find_peaks(pc, height=self.PEAK_HEIGHT_RELAXED,
                            distance=self.min_crossing_frames,
                            prominence=self.PEAK_PROMINENCE_RELAXED)
        cands = list(pks)
        if len(cands) >= 2:
            return self._process_crossings(cands, pc, traj, n)
        diff = self.debug_data["diff"]
        diff_smooth = gaussian_filter1d(diff, max(1, self.smoothing_window / 4))
        signs = np.sign(diff_smooth)
        zero_crossings = []
        for i in range(1, len(signs)):
            if signs[i] != signs[i-1] and signs[i] != 0:
                zero_crossings.append(i)
        if len(zero_crossings) >= 2:
            filtered = [zero_crossings[0]]
            for zc in zero_crossings[1:]:
                if zc - filtered[-1] >= self.min_crossing_frames:
                    filtered.append(zc)
            if len(filtered) >= 2:
                boost_sigma = max(2, self.min_crossing_frames * 0.1)
                for zc in filtered:
                    self._add_gaussian_boost(pc, zc, 0.5, boost_sigma)
                pks, _ = find_peaks(pc, height=self.PEAK_HEIGHT_RELAXED,
                                    distance=self.min_crossing_frames,
                                    prominence=0.01)
                cands = list(pks)
                if len(cands) >= 2:
                    return self._process_crossings(cands, pc, traj, n)
        abs_diff = np.abs(diff)
        abs_diff_smooth = gaussian_filter1d(abs_diff, max(1, self.smoothing_window / 4))
        ad_peaks, _ = find_peaks(abs_diff_smooth, distance=self.min_crossing_frames, prominence=0.001)
        if len(ad_peaks) >= 2:
            crossing_estimates = []
            for i in range(len(ad_peaks) - 1):
                mid = (ad_peaks[i] + ad_peaks[i + 1]) // 2
                crossing_estimates.append(mid)
            if len(crossing_estimates) >= 2:
                boost_sigma = max(2, self.min_crossing_frames * 0.1)
                for ce in crossing_estimates:
                    self._add_gaussian_boost(pc, ce, 0.5, boost_sigma)
                pks, _ = find_peaks(pc, height=self.PEAK_HEIGHT_RELAXED,
                                    distance=self.min_crossing_frames,
                                    prominence=0.01)
                cands = list(pks)
                if len(cands) >= 2:
                    return self._process_crossings(cands, pc, traj, n)
        return [], RhythmModel("T1", 0, None, 0, 0, 0)

    def _process_crossings(self, cands, pc, traj, n):
        tm = np.median(np.diff(cands))
        bps = [0]
        for i, iv in enumerate(np.diff(cands)):
            if iv > self.SEGMENT_BREAK_RATIO * tm:
                bps.append(cands[i] + int(iv) // 2)
        bps.append(n)
        all_c = []
        for si in range(len(bps) - 1):
            ss, se = bps[si], bps[si + 1]
            sc = [c for c in cands if ss <= c < se]
            if len(sc) < 2:
                all_c.extend(sc)
                continue
            fA, fB = sc[0::2], sc[1::2]
            tA = np.median(np.diff(fA)) if len(fA) >= 2 else tm * 2
            tB = np.median(np.diff(fB)) if len(fB) >= 2 else tm * 2
            sl = se - ss
            ni = 10 if sl <= 200 else (7 if sl <= 500 else 5)
            pseg = pc[ss:se].copy()
            tp_seg = np.sum(pseg)
            lfA = [c - ss for c in fA]
            lfB = [c - ss for c in fB]
            for _ in range(ni):
                pr = np.zeros(len(pseg))
                if len(lfA) >= 2:
                    pr += self._family_reinforcement(pseg, lfA, tA, self.REINFORCEMENT_MAX_CONTRIBUTION)
                if len(lfB) >= 2:
                    pr += self._family_reinforcement(pseg, lfB, tB, self.REINFORCEMENT_MAX_CONTRIBUTION)
                pn = pseg + pr
                tn = np.sum(pn)
                if tn > 0:
                    pn = pn * (tp_seg / tn)
                pseg = pn
                lp, _ = find_peaks(pseg, height=self.PEAK_HEIGHT_STRICT,
                                   distance=self.min_crossing_frames,
                                   prominence=self.PEAK_PROMINENCE_SEGMENT)
                lc = list(lp)
                lfA, lfB = lc[0::2], lc[1::2]
            all_c.extend([c + ss for c in lc])
            pc[ss:se] = pseg
        cands = sorted(set(all_c))
        tm = np.median(np.diff(cands)) if len(cands) >= 2 else self.min_crossing_frames * 2
        rm = RhythmModel("T1", tm, None, 0, tm * self.rhythm_sigma_ratio, cands[0] if cands else 0)
        diff = self.debug_data["diff"]
        ivs = []
        for i in range(len(cands) - 1):
            s, e = cands[i], cands[i+1]
            ivs.append(CrossingInterval(s, e, min(np.mean(np.abs(diff[s:e])) * 10, 1.0)))
        return ivs, rm

    def _add_gaussian_boost(self, pr, pos, boost, sigma):
        n = len(pr)
        if pos < 0 or pos >= n:
            return
        pi = int(pos)
        sr = int(3 * sigma)
        s, e = max(0, pi - sr), min(n, pi + sr + 1)
        fr = np.arange(s, e)
        pr[s:e] += boost * np.exp(-0.5 * ((fr - pos) / sigma)**2)

    def _family_reinforcement(self, pc, fc, tc, mc=0.1):
        n = len(pc)
        pr = np.zeros(n)
        if len(fc) < 2:
            return pr
        sigma = tc * self.rhythm_sigma_ratio
        for i in range(len(fc) - 1):
            x, y = fc[i], fc[i+1]
            T = y - x
            bb = pc[x] * pc[y]
            k = 1
            while k <= self.REINFORCEMENT_MAX_HOPS:
                pos = y + k * T
                if pos >= n:
                    break
                self._add_gaussian_boost(pr, pos, bb * self.REINFORCEMENT_DECAY**k, sigma)
                k += 1
            k = 1
            while k <= self.REINFORCEMENT_MAX_HOPS:
                pos = x - k * T
                if pos < 0:
                    break
                self._add_gaussian_boost(pr, pos, bb * self.REINFORCEMENT_DECAY**k, sigma)
                k += 1
        pm = np.max(pc)
        if pm > 0:
            rm_val = np.max(pr)
            if rm_val > 0:
                pr = pr * (pm * mc / rm_val)
        return pr

    # =======================================================================
    # Feature extraction (V1 + BIS ratio features)
    # =======================================================================

    def _extract_biomechanical_features(self, af):
        n = len(af)
        ap = self._ap_axis
        vert = self._vert_axis
        direction = self._direction
        features = {}
        px = np.zeros(n)
        for i, f in enumerate(af):
            lp = f.landmark_positions
            if lp:
                lh = lp.get("left_hip", (0, 0, 0))
                rh = lp.get("right_hip", (0, 0, 0))
                px[i] = (lh[ap] + rh[ap]) / 2.0
        thigh_lengths = []
        for f in af[::max(1, n//20)]:
            lp = f.landmark_positions
            if lp:
                for side in ["left", "right"]:
                    hip = np.array(lp.get(f"{side}_hip", (0, 0, 0)))
                    knee = np.array(lp.get(f"{side}_knee", (0, 0, 0)))
                    d = np.linalg.norm(hip - knee)
                    if d > 50:
                        thigh_lengths.append(d)
        thigh_len = np.mean(thigh_lengths) if thigh_lengths else 400.0
        for side in ["left", "right"]:
            knee_ap = np.zeros(n)
            heel_ap = np.zeros(n)
            ankle_ap = np.zeros(n)
            heel_z = np.zeros(n)
            toe_z = np.zeros(n)
            ankle_z = np.zeros(n)
            knee_angle = np.zeros(n)
            ankle_angle = np.zeros(n)
            has_heel = False
            has_toe = False
            has_ankle = False
            for i, f in enumerate(af):
                lp = f.landmark_positions
                if not lp:
                    continue
                knee_pos = lp.get(f"{side}_knee", None)
                heel_pos = lp.get(f"{side}_heel", None)
                ankle_pos = lp.get(f"{side}_ankle", None)
                toe_pos = lp.get(f"{side}_toe", None)
                if knee_pos:
                    knee_ap[i] = knee_pos[ap]
                if heel_pos:
                    heel_ap[i] = heel_pos[ap]
                    heel_z[i] = heel_pos[vert]
                    has_heel = True
                if ankle_pos:
                    ankle_ap[i] = ankle_pos[ap]
                    ankle_z[i] = ankle_pos[vert]
                    has_ankle = True
                if toe_pos:
                    toe_z[i] = toe_pos[vert]
                    has_toe = True
                if side == "left":
                    knee_angle[i] = f.left_knee_angle
                    ankle_angle[i] = f.left_ankle_angle
                else:
                    knee_angle[i] = f.right_knee_angle
                    ankle_angle[i] = f.right_ankle_angle

            # --- V1 features (unchanged) ---
            if has_heel:
                knee_heel_ap_raw = (heel_ap - knee_ap) * direction
            else:
                knee_heel_ap_raw = (ankle_ap - knee_ap) * direction
            knee_heel_ap = knee_heel_ap_raw / thigh_len
            sigma_s = max(1.0, self.smoothing_window / 6.0)
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
                heel_vert_norm = np.zeros(n)
                heel_vert_vel = np.zeros(n)
            if has_toe:
                toe_z_smooth = gaussian_filter1d(toe_z, sigma_s)
                tz_min, tz_max = np.min(toe_z_smooth), np.max(toe_z_smooth)
                tz_range = tz_max - tz_min if tz_max > tz_min else 1.0
                toe_vert_norm = (toe_z_smooth - tz_min) / tz_range
                toe_vert_vel = np.gradient(toe_vert_norm) * self.fps
            else:
                toe_vert_norm = np.zeros(n)
                toe_vert_vel = np.zeros(n)
            knee_angle_smooth = gaussian_filter1d(knee_angle, sigma_s)
            knee_angle_vel = np.gradient(knee_angle_smooth) * self.fps

            # --- NEW: BIS ratio features ---
            # omega_knee = knee angular velocity (deg/s) - already computed
            omega_knee = knee_angle_vel  # deg/s

            # omega_ankle = ankle angular velocity (deg/s)
            ankle_angle_smooth = gaussian_filter1d(ankle_angle, sigma_s)
            omega_ankle = np.gradient(ankle_angle_smooth) * self.fps

            # v_toe_vert = vertical velocity of toe (mm/s)
            if has_toe:
                toe_z_raw_smooth = gaussian_filter1d(toe_z, sigma_s)
                v_toe_vert = np.gradient(toe_z_raw_smooth) * self.fps
            else:
                v_toe_vert = np.zeros(n)

            # v_ankle_vert = vertical velocity of ankle (mm/s)
            if has_ankle:
                ankle_z_smooth = gaussian_filter1d(ankle_z, sigma_s)
                v_ankle_vert = np.gradient(ankle_z_smooth) * self.fps
            else:
                v_ankle_vert = np.zeros(n)

            # Compute ratio features using absolute values
            abs_omega_knee = np.abs(omega_knee)
            abs_v_toe = np.abs(v_toe_vert)
            abs_v_ankle = np.abs(v_ankle_vert)
            denom = abs_omega_knee + abs_v_toe + abs_v_ankle + 1e-10

            R_toe = abs_v_toe / denom
            R_knee = abs_omega_knee / denom

            # delta_omega = omega_knee - omega_ankle (deg/s)
            delta_omega = omega_knee - omega_ankle

            # --- V2: Knee velocity intensity ---
            # |omega_knee| normalized by local maximum, bounded [0, 1].
            # At TO: knee is moving fast => knee_vel_norm ~ 0.7
            # At HS: knee is nearly stopped => knee_vel_norm ~ 0.2-0.3
            # Sign-convention-independent (uses absolute value).
            # Cohen's d: 1.5-3.2 across Nature/PD/Fukuchi datasets.
            win_kvi = max(int(self.fps), 30)  # ~1 second window
            local_max_knee = np.ones(n)
            for idx in range(n):
                ws = max(0, idx - win_kvi // 2)
                we = min(n, idx + win_kvi // 2 + 1)
                local_max_knee[idx] = np.max(abs_omega_knee[ws:we])
            knee_vel_norm = abs_omega_knee / (local_max_knee + 1e-6)

            # Velocity confidence: when total velocity is very low
            # (e.g. freezing of gait), ratios become noisy.
            # Compute a per-frame confidence that scales the weight of ratio features.
            vel_magnitude = abs_omega_knee + abs_v_toe + abs_v_ankle
            vel_med = np.median(vel_magnitude[vel_magnitude > 0]) if np.any(vel_magnitude > 0) else 1.0
            vel_threshold = vel_med * 0.25
            # Confidence goes from 0 (unreliable) to 1 (reliable)
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
                # --- NEW ---
                "R_toe": R_toe,
                "R_knee": R_knee,
                "delta_omega": delta_omega,
                "vel_confidence": vel_confidence,
                "has_ankle": has_ankle,
                "knee_vel_norm": knee_vel_norm,
            }
        return features

    # =======================================================================
    # Log-likelihood computation (V1 + BIS features)
    # =======================================================================

    def _compute_log_likelihood_hs_array(self, features_side, start, end):
        f = features_side
        s, e = start, end
        ll = np.zeros(e - s)
        ll += HS_PRIORS["knee_heel_ap"].log_prob_array(f["knee_heel_ap"][s:e])
        ll += HS_PRIORS["ankle_ap_vel"].log_prob_array(f["ankle_ap_vel"][s:e])
        if f["has_heel"]:
            ll += HS_PRIORS["heel_vert"].log_prob_array(f["heel_vert"][s:e])
            ll += HS_PRIORS["heel_vert_vel"].log_prob_array(f["heel_vert_vel"][s:e])
        ll += HS_PRIORS["knee_angle"].log_prob_array(f["knee_angle"][s:e])
        ll += HS_PRIORS["knee_angle_vel"].log_prob_array(f["knee_angle_vel"][s:e])
        # --- NEW BIS features (weighted by velocity confidence) ---
        if f.get("has_toe", False) or f.get("has_ankle", False):
            vc = f["vel_confidence"][s:e]
            ll += vc * HS_PRIORS["R_toe"].log_prob_array(f["R_toe"][s:e])
            ll += vc * HS_PRIORS["R_knee"].log_prob_array(f["R_knee"][s:e])
            ll += vc * HS_PRIORS["delta_omega"].log_prob_array(f["delta_omega"][s:e])
            # V2: normalized knee angular velocity
            if "knee_vel_norm" in f:
                ll += vc * HS_PRIORS["knee_vel_norm"].log_prob_array(f["knee_vel_norm"][s:e])
        return ll

    def _compute_log_likelihood_to_array(self, features_side, start, end):
        f = features_side
        s, e = start, end
        ll = np.zeros(e - s)
        ll += TO_PRIORS["knee_heel_ap"].log_prob_array(f["knee_heel_ap"][s:e])
        ll += TO_PRIORS["ankle_ap_vel"].log_prob_array(f["ankle_ap_vel"][s:e])
        if f["has_toe"]:
            ll += TO_PRIORS["toe_vert"].log_prob_array(f["toe_vert"][s:e])
            ll += TO_PRIORS["toe_vert_vel"].log_prob_array(f["toe_vert_vel"][s:e])
        ll += TO_PRIORS["knee_angle"].log_prob_array(f["knee_angle"][s:e])
        ll += TO_PRIORS["knee_angle_vel"].log_prob_array(f["knee_angle_vel"][s:e])
        # --- NEW BIS features (weighted by velocity confidence) ---
        if f.get("has_toe", False) or f.get("has_ankle", False):
            vc = f["vel_confidence"][s:e]
            ll += vc * TO_PRIORS["R_toe"].log_prob_array(f["R_toe"][s:e])
            ll += vc * TO_PRIORS["R_knee"].log_prob_array(f["R_knee"][s:e])
            ll += vc * TO_PRIORS["delta_omega"].log_prob_array(f["delta_omega"][s:e])
            # V2: normalized knee angular velocity
            if "knee_vel_norm" in f:
                ll += vc * TO_PRIORS["knee_vel_norm"].log_prob_array(f["knee_vel_norm"][s:e])
        return ll

    def _bayesian_event_assignment(self, crossings, features, af):
        n_intervals = len(crossings)
        if n_intervals == 0:
            return []
        INF = -1e9
        dp = [{} for _ in range(n_intervals)]
        bt = [{} for _ in range(n_intervals)]
        iv = crossings[0]
        s, e = iv.start_frame, iv.end_frame
        for sl in ["TO", "HS"]:
            for sr in ["TO", "HS"]:
                if sl == sr:
                    continue
                sk = (sl, sr)
                sc, fl, fr = self._best_bayesian_frames(s, e, sl, sr, features)
                dp[0][sk] = sc
                bt[0][sk] = (fl, fr, None)
        for i in range(1, n_intervals):
            iv = crossings[i]
            s, e = iv.start_frame, iv.end_frame
            for cl in ["TO", "HS"]:
                for cr in ["TO", "HS"]:
                    if cl == cr:
                        continue
                    ck = (cl, cr)
                    bps = INF
                    bpst = None
                    bf = None
                    for pl in ["TO", "HS"]:
                        for pr_s in ["TO", "HS"]:
                            pk = (pl, pr_s)
                            if pk not in dp[i-1] or pl == cl or pr_s == cr:
                                continue
                            si_val, fl, fr = self._best_bayesian_frames(s, e, cl, cr, features)
                            ts = dp[i-1][pk] + si_val
                            if ts > bps:
                                bps, bpst, bf = ts, pk, (fl, fr)
                    if bpst is not None:
                        dp[i][ck] = bps
                        bt[i][ck] = (bf[0], bf[1], bpst)
        bfs = INF
        bfst = None
        for sk in dp[n_intervals-1]:
            if dp[n_intervals-1][sk] > bfs:
                bfs, bfst = dp[n_intervals-1][sk], sk
        if bfst is None:
            return []
        events = []
        cs = bfst
        for i in range(n_intervals-1, -1, -1):
            fl, fr, ps = bt[i][cs]
            sl, sr = cs
            etl = "heel_strike" if sl == "HS" else "toe_off"
            etr = "heel_strike" if sr == "HS" else "toe_off"
            events.append(GaitEvent(fl, fl/self.fps, etl, "left", 0.8))
            events.append(GaitEvent(fr, fr/self.fps, etr, "right", 0.8))
            if ps is not None:
                cs = ps
        events.sort(key=lambda e: e.frame_index)
        return events

    def _best_bayesian_frames(self, s, e, etl, etr, features):
        if e <= s:
            return -1e9, s, s
        interval_len = e - s
        phase = np.linspace(0, 1, interval_len)
        if etl == "HS":
            ll_left = self._compute_log_likelihood_hs_array(features["left"], s, e)
            tp_left = HS_PHASE_PRIOR.log_prob_array(phase)
        else:
            ll_left = self._compute_log_likelihood_to_array(features["left"], s, e)
            tp_left = TO_PHASE_PRIOR.log_prob_array(phase)
        if etr == "HS":
            ll_right = self._compute_log_likelihood_hs_array(features["right"], s, e)
            tp_right = HS_PHASE_PRIOR.log_prob_array(phase)
        else:
            ll_right = self._compute_log_likelihood_to_array(features["right"], s, e)
            tp_right = TO_PHASE_PRIOR.log_prob_array(phase)
        post_left = ll_left + tp_left
        post_right = ll_right + tp_right
        if len(post_left) == 0 or len(post_right) == 0:
            return -1e9, s, s
        il = np.argmax(post_left)
        ir = np.argmax(post_right)
        score = post_left[il] + post_right[ir]
        return score, s + il, s + ir

    def _compute_bayesian_event_probs(self, features, af):
        n = len(af)
        probs = {}
        for side in ["left", "right"]:
            f = features[side]
            ll_hs = self._compute_log_likelihood_hs_array(f, 0, n)
            ll_to = self._compute_log_likelihood_to_array(f, 0, n)
            max_ll = np.maximum(ll_hs, ll_to)
            p_hs = np.exp(ll_hs - max_ll)
            p_to = np.exp(ll_to - max_ll)
            total = p_hs + p_to + 1e-10
            p_hs = p_hs / total
            p_to = p_to / total
            sigma_s = max(1.0, self.smoothing_window / 6.0)
            p_hs = gaussian_filter1d(p_hs, sigma_s)
            p_to = gaussian_filter1d(p_to, sigma_s)
            probs[f"{side}_hs"] = np.clip(p_hs, 0.01, 0.99)
            probs[f"{side}_to"] = np.clip(p_to, 0.01, 0.99)
        return probs

    def _event_probs_angles(self, af):
        n = len(af)
        lkr = np.array([f.left_knee_angle for f in af])
        rkr = np.array([f.right_knee_angle for f in af])
        lhr = np.array([f.left_hip_angle for f in af])
        rhr = np.array([f.right_hip_angle for f in af])
        lk = self._normalize_angles(lkr, 0, 60)
        rk = self._normalize_angles(rkr, 0, 60)
        lh = self._normalize_angles(lhr, 0, 20)
        rh = self._normalize_angles(rhr, 0, 20)
        lkv = np.gradient(lk)
        rkv = np.gradient(rk)
        if n > 5:
            lkv = savgol_filter(lkv, 5, 2)
            rkv = savgol_filter(rkv, 5, 2)
        probs = {"left_hs": np.zeros(n), "left_to": np.zeros(n),
                 "right_hs": np.zeros(n), "right_to": np.zeros(n)}
        for side, knee, hip, kv in [("left", lk, lh, lkv), ("right", rk, rh, rkv)]:
            khs = 1.0 / (1.0 + np.exp(-0.1 * (-knee - (-5.0))))
            hhs = 1.0 / (1.0 + np.exp(-0.1 * (hip - 15.0)))
            probs[f"{side}_hs"] = 0.6 * khs + 0.4 * hhs
            kto = 1.0 / (1.0 + np.exp(-0.5 * (kv - 1.5)))
            ato = 1.0 / (1.0 + np.exp(-0.1 * (knee - 10.0)))
            probs[f"{side}_to"] = 0.7 * kto + 0.3 * ato
        for k in probs:
            probs[k] = np.clip(probs[k], 0.01, 0.99)
        return probs

    def _normalize_angles(self, a, tmin, tmax):
        mn, mx = np.min(a), np.max(a)
        if mx - mn < 1e-6:
            return np.full_like(a, (tmin + tmax) / 2)
        return (a - mn) / (mx - mn) * (tmax - tmin) + tmin

    def _viterbi_fallback(self, crossings, ep):
        ni = len(crossings)
        INF = -1e9
        dp = [{} for _ in range(ni)]
        bt = [{} for _ in range(ni)]
        iv = crossings[0]
        s, e = iv.start_frame, iv.end_frame
        for sl in ["TO", "HS"]:
            for sr in ["TO", "HS"]:
                if sl == sr:
                    continue
                sk = (sl, sr)
                sc, fl, fr = self._best_frames_ep(s, e, sl, sr, ep)
                dp[0][sk] = sc
                bt[0][sk] = (fl, fr, None)
        for i in range(1, ni):
            iv = crossings[i]
            s, e = iv.start_frame, iv.end_frame
            for cl in ["TO", "HS"]:
                for cr in ["TO", "HS"]:
                    if cl == cr:
                        continue
                    ck = (cl, cr)
                    bps = INF
                    bpst = None
                    bf = None
                    for pl in ["TO", "HS"]:
                        for pr_s in ["TO", "HS"]:
                            pk = (pl, pr_s)
                            if pk not in dp[i-1] or pl == cl or pr_s == cr:
                                continue
                            si_val, fl, fr = self._best_frames_ep(s, e, cl, cr, ep)
                            ts = dp[i-1][pk] + si_val
                            if ts > bps:
                                bps, bpst, bf = ts, pk, (fl, fr)
                    if bpst is not None:
                        dp[i][ck] = bps
                        bt[i][ck] = (bf[0], bf[1], bpst)
        bfs = INF
        bfst = None
        for sk in dp[ni-1]:
            if dp[ni-1][sk] > bfs:
                bfs, bfst = dp[ni-1][sk], sk
        if bfst is None:
            return []
        events = []
        cs = bfst
        for i in range(ni-1, -1, -1):
            fl, fr, ps = bt[i][cs]
            sl, sr = cs
            etl = "heel_strike" if sl == "HS" else "toe_off"
            etr = "heel_strike" if sr == "HS" else "toe_off"
            events.append(GaitEvent(fl, fl/self.fps, etl, "left", float(ep[f"left_{sl.lower()}"][fl])))
            events.append(GaitEvent(fr, fr/self.fps, etr, "right", float(ep[f"right_{sr.lower()}"][fr])))
            if ps is not None:
                cs = ps
        events.sort(key=lambda e: e.frame_index)
        return events

    def _best_frames_ep(self, s, e, etl, etr, ep):
        pl = ep[f"left_{etl.lower()}"][s:e]
        pr = ep[f"right_{etr.lower()}"][s:e]
        if len(pl) == 0 or len(pr) == 0:
            return -1e9, s, s
        il, ir = np.argmax(pl), np.argmax(pr)
        return np.log(pl[il]+1e-10) + np.log(pr[ir]+1e-10), s+il, s+ir

    def _refine_velocity_zerocrossing(self, events, af, event_type, rm):
        if not events or len(af) == 0:
            return events
        n = len(af)
        if af[0].landmark_positions is None or "left_ankle" not in af[0].landmark_positions:
            return events
        ap = self._ap_axis
        direction = self._direction
        px = np.array([(f.landmark_positions.get("left_hip", (0.5,0,0))[ap] +
                        f.landmark_positions.get("right_hip", (0.5,0,0))[ap]) / 2 for f in af])
        zs, zv = {}, {}
        for side in ["left", "right"]:
            ax = np.array([f.landmark_positions.get(f"{side}_ankle", (0.5,0,0))[ap] for f in af])
            rel = ax - px
            if direction < 0:
                rel = -rel
            ss = max(1, self.smoothing_window / 6)
            rs = gaussian_filter1d(rel, ss)
            vel = gaussian_filter1d(np.gradient(rs), ss * 0.5)
            zs[side], zv[side] = rs, vel
        T = rm.T1 if rm.T1 > 0 else self.min_crossing_frames * 2
        win = max(3, int(T / 3))
        win = min(win, int(T / 2))
        heel_z_smooth = {}
        toe_z_smooth = {}
        has_heel = ("left_heel" in af[0].landmark_positions)
        has_toe = ("left_toe" in af[0].landmark_positions)
        if has_heel and event_type == "hs":
            _, vert = detect_axes(af)
            for side in ["left", "right"]:
                hz = np.array([f.landmark_positions.get(f"{side}_heel", (0,0,0))[vert] for f in af])
                if np.ptp(hz) > self.VERTICAL_RANGE_THRESHOLD:
                    heel_z_smooth[side] = gaussian_filter1d(hz, 1.0)
        if has_toe and event_type == "to":
            _, vert = detect_axes(af)
            for side in ["left", "right"]:
                tz = np.array([f.landmark_positions.get(f"{side}_toe", (0,0,0))[vert] for f in af])
                if np.ptp(tz) > self.VERTICAL_RANGE_THRESHOLD:
                    toe_z_smooth[side] = gaussian_filter1d(tz, 1.0)
        max_vert_correction = max(self.VERTICAL_CORRECTION_MIN_SHIFT,
                                  min(int(T * self.VERTICAL_CORRECTION_MAX_RATIO), int(T / 6)))
        out = []
        for event in events:
            side = event.side if event.side in ["left", "right"] else "left"
            sig, vel = zs.get(side, zs["left"]), zv.get(side, zv["left"])
            f = event.frame_index
            s, e = max(0, f - win), min(n, f + win + 1)
            v_win = vel[s:e]
            z_win = sig[s:e]
            if len(z_win) < 3:
                out.append(event)
                continue
            vzc_frame = None
            center = f - s
            if event_type == "hs":
                best_dist = len(v_win)
                for i in range(1, len(v_win)):
                    if v_win[i-1] > 0 and v_win[i] <= 0:
                        dist = abs(i - center)
                        if dist < best_dist:
                            best_dist = dist
                            vzc_frame = s + i
            else:
                best_dist = len(v_win)
                for i in range(1, len(v_win)):
                    if v_win[i-1] < 0 and v_win[i] >= 0:
                        dist = abs(i - center)
                        if dist < best_dist:
                            best_dist = dist
                            vzc_frame = s + i
            if vzc_frame is not None:
                rf = vzc_frame
            else:
                if event_type == "hs":
                    rf = s + np.argmax(z_win)
                else:
                    rf = s + np.argmin(z_win)
            if event_type == "hs" and side in heel_z_smooth:
                heel_win = heel_z_smooth[side][s:e]
                if len(heel_win) > 0:
                    heel_min_local = s + np.argmin(heel_win)
                    shift = heel_min_local - rf
                    if self.VERTICAL_CORRECTION_MIN_SHIFT <= shift <= max_vert_correction:
                        correction = int(round(shift * self.VERTICAL_CORRECTION_SHRINKAGE))
                        rf = rf + correction
            if event_type == "to" and side in toe_z_smooth:
                toe_win = toe_z_smooth[side][s:e]
                if len(toe_win) > 0:
                    toe_min_local = s + np.argmin(toe_win)
                    shift = toe_min_local - rf
                    if self.VERTICAL_CORRECTION_MIN_SHIFT <= shift <= max_vert_correction:
                        correction = int(round(shift * self.VERTICAL_CORRECTION_SHRINKAGE))
                        rf = rf + correction
            rf = max(0, min(n - 1, rf))
            out.append(GaitEvent(rf, rf/self.fps, event.event_type, event.side, event.probability))
        return out

    def _boundary_events(self, crossings, ve, ep, nf, rm):
        be = []
        if not crossings or not ve:
            return be
        fc = crossings[0].start_frame
        lc = crossings[-1].end_frame
        T = rm.T1 if rm.T1 > 0 else self.min_crossing_frames * 2
        md = int(T / 4)
        ss, se = 0, fc
        if se > ss + md:
            for side in ["left", "right"]:
                for et in ["hs", "to"]:
                    pz = ep["%s_%s" % (side, et)][ss:se]
                    if len(pz) == 0:
                        continue
                    pks, _ = find_peaks(pz, height=self.BOUNDARY_PEAK_HEIGHT_LEADING,
                                        prominence=self.BOUNDARY_PEAK_PROMINENCE_LEADING,
                                        distance=md)
                    cs = list(pks)
                    if not cs and len(pz) > 5:
                        sp = pz[0]
                        fet = "heel_strike" if et == "hs" else "toe_off"
                        os_name = "right" if side == "left" else "left"
                        ts_near = any(ev.frame_index < T and ev.event_type == fet and ev.side == side for ev in ve)
                        os_very_near = any(ev.frame_index < T / 2 and ev.event_type == fet and ev.side == os_name for ev in ve)
                        mp = pz[len(pz)//2] if len(pz) > 10 else 0
                        is_sp = sp > mp + 0.1
                        leading_long_enough = len(pz) >= T / 3
                        if (sp > self.BOUNDARY_FALLBACK_LEADING_THRESHOLD
                                and is_sp and not ts_near and not os_very_near
                                and leading_long_enough):
                            ez = min(int(T / 4), len(pz))
                            mi = np.argmax(pz[:ez])
                            if self.BOUNDARY_MIN_LEADING_FRAME <= mi < T / 6:
                                cs = [mi]
                    if len(cs) > 1:
                        cs = [max(cs, key=lambda x: pz[x])]
                    for pk in cs:
                        bf = ss + pk
                        if not any(abs(bf - ev.frame_index) < md for ev in ve if ev.side == side):
                            fet = "heel_strike" if et == "hs" else "toe_off"
                            be.append(GaitEvent(bf, bf/self.fps, fet, side, float(pz[pk])))
        ss, se = lc, nf
        if se > ss + md:
            for side in ["left", "right"]:
                for et in ["hs", "to"]:
                    pz = ep["%s_%s" % (side, et)][ss:se]
                    if len(pz) == 0:
                        continue
                    pks, _ = find_peaks(pz, height=self.BOUNDARY_PEAK_HEIGHT_TRAILING,
                                        prominence=self.BOUNDARY_PEAK_PROMINENCE_TRAILING,
                                        distance=md)
                    cs = list(pks)
                    if not cs:
                        mi = np.argmax(pz)
                        trailing_len = se - ss
                        if trailing_len >= T / 2 and pz[mi] > self.BOUNDARY_FALLBACK_TRAILING_THRESHOLD:
                            cs = [mi]
                    if len(cs) > 1:
                        cs = [max(cs, key=lambda x: pz[x])]
                    for pk in cs:
                        bf = ss + pk
                        if not any(abs(bf - ev.frame_index) < md for ev in ve if ev.side == side):
                            fet = "heel_strike" if et == "hs" else "toe_off"
                            if et == "to" and (nf - bf) < T * self.TRAILING_TO_SUPPRESS_RATIO:
                                continue
                            be.append(GaitEvent(bf, bf/self.fps, fet, side, float(pz[pk])))
        return be

    def _gap_events(self, ee, ep, nf, rm):
        ge = []
        if len(ee) < 2:
            return ge
        ahs = sorted([ev for ev in ee if ev.event_type == "heel_strike"], key=lambda ev: ev.frame_index)
        if len(ahs) < 2:
            return ge
        ivs = np.diff([ev.frame_index for ev in ahs])
        tm = np.median(ivs)
        gt = tm * self.GAP_THRESHOLD_RATIO
        md = int(tm / 4)
        for side in ["left", "right"]:
            for ek, en in [("hs", "heel_strike"), ("to", "toe_off")]:
                se_list = sorted([ev for ev in ee if ev.event_type == en and ev.side == side], key=lambda ev: ev.frame_index)
                if len(se_list) < 2:
                    continue
                fs = [ev.frame_index for ev in se_list]
                for i in range(len(fs) - 1):
                    if fs[i+1] - fs[i] > gt:
                        ss_g, sse = fs[i] + md, fs[i+1] - md
                        if sse <= ss_g:
                            continue
                        pz = ep[f"{side}_{ek}"][ss_g:sse]
                        if len(pz) > 0:
                            pks, _ = find_peaks(pz, height=self.GAP_PEAK_HEIGHT,
                                                prominence=self.GAP_PEAK_PROMINENCE,
                                                distance=md)
                            for pk in pks:
                                f_idx = ss_g + pk
                                if not any(abs(f_idx - ev.frame_index) < md for ev in ee + ge if ev.side == side and ev.event_type == en):
                                    ge.append(GaitEvent(f_idx, f_idx/self.fps, en, side, float(pz[pk])))
        return ge

    def _build_cycles(self, hs, to):
        cycles = []
        for side in ["left", "right"]:
            shs = sorted([ev for ev in hs if ev.side == side], key=lambda x: x.frame_index)
            sto = sorted([ev for ev in to if ev.side == side], key=lambda x: x.frame_index)
            for i in range(len(shs) - 1):
                sh, eh = shs[i], shs[i+1]
                toe = None
                for t in sto:
                    if sh.frame_index < t.frame_index < eh.frame_index:
                        toe = t
                        break
                cd = (eh.frame_index - sh.frame_index) / self.fps
                if toe:
                    sd = (toe.frame_index - sh.frame_index) / self.fps
                    swd = cd - sd
                    sp = sd / cd * 100
                    swp = 100 - sp
                else:
                    sd = swd = sp = swp = None
                cycles.append(GaitCycle(len(cycles), side, sh.frame_index,
                                        toe.frame_index if toe else None, eh.frame_index,
                                        sh.time, toe.time if toe else None, eh.time,
                                        cd, sd, swd, sp, swp))
        cycles.sort(key=lambda c: c.start_frame)
        return cycles

    def _event_probs_zeni(self, af):
        n = len(af)
        ap = getattr(self, '_ap_axis', 0)
        direction = getattr(self, '_direction', 1)
        px = np.array([(f.landmark_positions.get('left_hip', (0.5,0,0))[ap] +
                        f.landmark_positions.get('right_hip', (0.5,0,0))[ap]) / 2 for f in af])
        lar = np.array([f.landmark_positions.get('left_ankle', (0.5,0,0))[ap] - px[i] for i, f in enumerate(af)])
        rar = np.array([f.landmark_positions.get('right_ankle', (0.5,0,0))[ap] - px[i] for i, f in enumerate(af)])
        if direction < 0:
            lar, rar = -lar, -rar
        if n > self.smoothing_window:
            lar = gaussian_filter1d(lar, self.smoothing_window / 4)
            rar = gaussian_filter1d(rar, self.smoothing_window / 4)
        lv, rv = np.gradient(lar), np.gradient(rar)
        probs = {'left_hs': np.zeros(n), 'left_to': np.zeros(n),
                 'right_hs': np.zeros(n), 'right_to': np.zeros(n)}
        for side, ar, v in [('left', lar, lv), ('right', rar, rv)]:
            mn, mx = np.min(ar), np.max(ar)
            r = mx - mn if mx > mn else 1
            pn = (ar - mn) / r
            va = np.abs(v)
            vm = np.max(va) if np.max(va) > 0 else 1
            vn = 1 - va / vm
            probs[side + '_hs'] = pn**2 * vn**0.5
            probs[side + '_to'] = (1-pn)**2 * vn**0.5
        for k in probs:
            pm = np.max(probs[k])
            if pm > 0:
                probs[k] = probs[k] / pm * 0.99
            probs[k] = np.clip(probs[k], 0.01, 0.99)
        return probs

    def _reinforce_event_probs(self, probs, niter=2, mc=0.1):
        EVENT_PROB_PEAK_HEIGHT = 0.3
        EVENT_PROB_PEAK_PROMINENCE = 0.05
        out = {}
        for key in probs:
            pc = probs[key].copy()
            tpi = np.sum(pc)
            pks, _ = find_peaks(pc, height=EVENT_PROB_PEAK_HEIGHT,
                                distance=self.min_crossing_frames,
                                prominence=EVENT_PROB_PEAK_PROMINENCE)
            if len(pks) < 2:
                out[key] = pc
                continue
            tm = np.median(np.diff(pks))
            for _ in range(niter):
                pks, _ = find_peaks(pc, height=EVENT_PROB_PEAK_HEIGHT,
                                    distance=self.min_crossing_frames,
                                    prominence=EVENT_PROB_PEAK_PROMINENCE)
                cands = list(pks)
                if len(cands) < 2:
                    break
                sigma = tm * self.rhythm_sigma_ratio
                pr = self._single_scale_reinforcement(pc, cands, sigma)
                pm = np.max(pc)
                rm_val = np.max(pr)
                if rm_val > 0:
                    pr = pr * (pm * mc / rm_val)
                pn = pc + pr
                tn = np.sum(pn)
                if tn > 0:
                    pn = pn * (tpi / tn)
                pc = pn
            out[key] = np.clip(pc, 0.01, 0.99)
        return out

    def _single_scale_reinforcement(self, pc, cands, sigma):
        n = len(pc)
        pr = np.zeros(n)
        for i in range(len(cands)):
            for d in range(1, min(4, len(cands) - i)):
                x, y = cands[i], cands[i + d]
                T = y - x
                bb = pc[x] * pc[y]
                k = 2
                while x + k * T < n:
                    self._add_gaussian_boost(pr, x + k * T, bb * self.REINFORCEMENT_DECAY**(k-1), sigma)
                    k += 1
                k = 1
                while x - k * T >= 0:
                    self._add_gaussian_boost(pr, x - k * T, bb * self.REINFORCEMENT_DECAY**(k-1), sigma)
                    k += 1
        return pr

    def _refine_boundary_bayesian(self, boundary_events, features, crossings, rm):
        """Refine boundary events using Bayesian biomechanical features."""
        T = rm.T1 if rm.T1 > 0 else self.min_crossing_frames * 2
        win = max(3, int(T / 4))
        n = len(features['left']['knee_heel_ap'])

        refined = []
        for event in boundary_events:
            side = event.side
            f = event.frame_index
            s = max(0, f - win)
            e = min(n, f + win + 1)

            if e - s < 3:
                refined.append(event)
                continue

            feat = features[side]

            if event.event_type == 'heel_strike':
                ll = self._compute_log_likelihood_hs_array(feat, s, e)
            else:
                ll = self._compute_log_likelihood_to_array(feat, s, e)

            # Find the best frame near the boundary estimate
            best_idx = np.argmax(ll)
            best_frame = s + best_idx

            # Only shift if the new frame is within win/2 of original
            if abs(best_frame - f) <= win // 2:
                refined.append(GaitEvent(best_frame, best_frame/self.fps,
                                        event.event_type, event.side, event.probability))
            else:
                refined.append(event)

        return refined

    def _refine_to_with_toe_vertical(self, events, af, rm):
        """Refine TO timing using toe vertical velocity."""
        if not events or len(af) == 0:
            return events
        n = len(af)
        vert = self._vert_axis

        has_toe = ('left_toe' in af[0].landmark_positions if af[0].landmark_positions else False)
        if not has_toe:
            return events

        T = rm.T1 if rm.T1 > 0 else self.min_crossing_frames * 2
        max_shift = max(2, int(T / 10))

        sigma_s = max(1.0, self.smoothing_window / 6.0)
        toe_z_vel = {}
        for side in ['left', 'right']:
            tz = np.array([f.landmark_positions.get(side + '_toe', (0,0,0))[vert]
                          for f in af])
            tz_smooth = gaussian_filter1d(tz, sigma_s)
            tz_vel = np.gradient(tz_smooth) * self.fps
            toe_z_vel[side] = tz_vel

        out = []
        for event in events:
            side = event.side if event.side in ['left', 'right'] else 'left'
            f = event.frame_index

            s = f
            e = min(n, f + max_shift + 1)

            if e - s < 2:
                out.append(event)
                continue

            tv = toe_z_vel[side][s:e]

            best_idx = np.argmax(tv)
            best_vel = tv[best_idx]

            if best_vel > 0 and best_idx > 0:
                rf = s + best_idx
                rf = min(n - 1, rf)
            else:
                rf = f

            out.append(GaitEvent(rf, rf/self.fps, event.event_type, event.side, event.probability))

        return out
