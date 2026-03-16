#!/usr/bin/env python3
"""Benchmark markerless video gait event detection on BioCV.

Pipeline par trial × caméra × condition de dégradation :
  1. Charger les GT events du C3D (via BioCVExtractor)
  2. Charger ou extraire le JSON MediaPipe (200fps natif)
  3. Pour chaque condition de dégradation (7 niveaux) :
     a. Sous-échantillonner les frames (FPS)
     b. Pour résolution/contraste/perspective : charger un JSON pré-dégradé
     c. Normaliser + calculer les angles
     d. Exécuter les 10 détecteurs gaitkit
     e. Évaluer vs GT à 25/50/100ms
  4. Écrire le CSV + afficher le résumé (F1 moyen, MAE moyen)

Conditions de dégradation :
  1. lab_reference      : 200fps, 1080p, contrast=1.0, perspective=0.0
  2. clinique_pro       : 100fps, 1080p, contrast=1.0, perspective=0.0
  3. smartphone_trepied :  60fps, 1080p, contrast=1.0, perspective=0.0
  4. smartphone_main    :  30fps, 1080p, contrast=1.0, perspective=0.10
  5. smartphone_basique :  30fps,  720p, contrast=0.8, perspective=0.10
  6. teleconsultation   :  30fps,  480p, contrast=0.7, perspective=0.15
  7. couloir_hopital    :  15fps,  480p, contrast=0.6, perspective=0.20

Usage:
    python run_markerless_benchmark.py \\
        --data-dir /path/to/BATH-01258 \\
        --output benchmark_markerless.csv \\
        [--subjects P03 P04] \\
        [--max-trials 5] \\
        [--conditions lab_reference clinique_pro] \\
        [--skip-extraction]

Sortie CSV (une ligne par trial × caméra × condition × détecteur × tolérance):
    subject, trial, camera, condition, fps_video, detector, tolerance_ms,
    n_gt_hs, n_det_hs, tp_hs, fp_hs, fn_hs, precision_hs, recall_hs, f1_hs, mae_hs_ms,
    n_gt_to, n_det_to, tp_to, fp_to, fn_to, precision_to, recall_to, f1_to, mae_to_ms
"""

import argparse
import csv
import json
import copy
import subprocess
import sys
import traceback
from pathlib import Path
from collections import defaultdict
import numpy as np


# ── Conditions de dégradation ────────────────────────────────────────────────

DEGRADATION_CONDITIONS = [
    {
        "name": "lab_reference",
        "label": "Lab référence",
        "fps": 200, "width": 1920, "height": 1080,
        "contrast": 1.0, "perspective": 0.0,
        "needs_video_degrade": False,
    },
    {
        "name": "clinique_pro",
        "label": "Clinique pro",
        "fps": 100, "width": 1920, "height": 1080,
        "contrast": 1.0, "perspective": 0.0,
        "needs_video_degrade": False,
    },
    {
        "name": "smartphone_trepied",
        "label": "Smartphone trépied",
        "fps": 60, "width": 1920, "height": 1080,
        "contrast": 1.0, "perspective": 0.0,
        "needs_video_degrade": False,
    },
    {
        "name": "smartphone_main",
        "label": "Smartphone main",
        "fps": 30, "width": 1920, "height": 1080,
        "contrast": 1.0, "perspective": 0.10,
        "needs_video_degrade": True,
    },
    {
        "name": "smartphone_basique",
        "label": "Smartphone basique",
        "fps": 30, "width": 1280, "height": 720,
        "contrast": 0.8, "perspective": 0.10,
        "needs_video_degrade": True,
    },
    {
        "name": "teleconsultation",
        "label": "Téléconsultation",
        "fps": 30, "width": 854, "height": 480,
        "contrast": 0.7, "perspective": 0.15,
        "needs_video_degrade": True,
    },
    {
        "name": "couloir_hopital",
        "label": "Couloir hôpital",
        "fps": 15, "width": 854, "height": 480,
        "contrast": 0.6, "perspective": 0.20,
        "needs_video_degrade": True,
    },
]

COND_BY_NAME = {c["name"]: c for c in DEGRADATION_CONDITIONS}

# ── Détecteurs ───────────────────────────────────────────────────────────────

DETECTORS = [
    "gk_bike", "gk_zeni", "gk_oconnor", "gk_hreljac",
    "gk_mickelborough", "gk_ghoussayni", "gk_vancanneyt",
    "gk_dgei", "gk_intellevent", "gk_deepevent",
]

TOLERANCES_MS = [25, 50, 100]

# Caméras latérales BioCV
LATERAL_CAMERAS = ["01", "05"]


# ── Découverte des trials ────────────────────────────────────────────────────

def find_trials(data_dir, subjects=None, max_trials=None):
    """Découvre les trials WALK avec vidéos latérales."""
    data_dir = Path(data_dir)
    trials = []

    for subj_dir in sorted(data_dir.glob("P*")):
        if not subj_dir.is_dir():
            continue
        subj = subj_dir.name
        if subjects and subj not in subjects:
            continue

        walk_dirs = sorted(subj_dir.glob("{}_WALK_*".format(subj)))
        if max_trials:
            walk_dirs = walk_dirs[:max_trials]

        for trial_dir in walk_dirs:
            trial = trial_dir.name
            c3d_path = trial_dir / "markers.c3d"
            if not c3d_path.exists():
                continue

            for cam in LATERAL_CAMERAS:
                video_path = trial_dir / "{}.mp4".format(cam)
                if video_path.exists():
                    trials.append({
                        "subject": subj,
                        "trial": trial,
                        "camera": cam,
                        "video_path": str(video_path),
                        "c3d_path": str(c3d_path),
                    })

    return trials


# ── Ground truth ─────────────────────────────────────────────────────────────

def get_gt_events(c3d_path):
    """Extrait les GT events du C3D via BioCVExtractor."""
    from gaitkit.extractors.biocv_extractor import BioCVExtractor

    trial_dir = Path(c3d_path).parent
    subj_dir = trial_dir.parent
    data_dir = subj_dir.parent

    ext = BioCVExtractor(str(data_dir))
    result = ext.extract_file(Path(c3d_path))

    fps = result.fps
    gt = result.ground_truth

    vr = gt.valid_frame_range
    vr_start = vr[0] if vr else 0
    vr_end = vr[1] if vr else result.n_frames

    # Combiner L+R, filtrer au valid range, convertir en secondes
    all_hs = []
    for side in ("left", "right"):
        for f in gt.hs_frames.get(side, []):
            if vr_start <= f <= vr_end:
                all_hs.append(f / fps)
    all_to = []
    for side in ("left", "right"):
        for f in gt.to_frames.get(side, []):
            if vr_start <= f <= vr_end:
                all_to.append(f / fps)

    return {
        "hs": sorted(all_hs),
        "to": sorted(all_to),
        "fps": fps,
        "t_start": vr_start / fps,
        "t_end": vr_end / fps,
        "valid_range": (vr_start, vr_end),
    }


# ── Extraction et dégradation vidéo ─────────────────────────────────────────

def extract_video(video_path, output_json):
    """Extrait les landmarks MediaPipe via myogait."""
    import myogait

    print("    Extracting: {} ...".format(video_path), end=" ", flush=True)
    try:
        data = myogait.extract(video_path, correct_inversions=True)
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(data, f)
        print("OK ({} frames)".format(len(data.get("frames", []))))
        return data
    except Exception as e:
        print("FAILED: {}".format(e))
        return None


def degrade_video(video_path, output_path, condition):
    """Dégrade une vidéo via ffmpeg (résolution, fps, contraste, perspective).

    Retourne True si succès, False sinon.
    """
    cond = condition
    filters = []

    # Perspective (trapèze léger via pad + perspective filter)
    persp = cond["perspective"]
    if persp > 0:
        w, h = 1920, 1080
        dx = int(w * persp * 0.5)
        dy = int(h * persp * 0.3)
        filters.append(
            "perspective={}:{}:{}:{}:{}:{}:{}:{}"
            ":sense=destination:eval=init".format(
                dx, dy, w - dx, dy, 0, h, w, h))

    # Contraste
    if cond["contrast"] < 1.0:
        filters.append("eq=contrast={}".format(cond["contrast"]))

    # Résolution
    filters.append("scale={}:{}".format(cond["width"], cond["height"]))

    vf = ",".join(filters) if filters else None

    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-r", str(cond["fps"])]
    cmd += ["-an", str(output_path)]

    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        return True
    except Exception as e:
        print("    ffmpeg failed: {}".format(e))
        return False


def get_data_for_condition(trial_info, condition, json_dir, skip_extraction):
    """Retourne les données pour une condition de dégradation.

    - Conditions 1-3 (FPS only) : charge le JSON extrait au FPS natif
      Convention : {trial}_cam{cam}_{fps}fps.json
    - Conditions 4-7 (vidéo dégradée) : charge un JSON pré-dégradé ou
      dégrade la vidéo + extrait
      Convention : {condition_name}/{trial}_cam{cam}.json
    """
    cond = condition
    trial = trial_info["trial"]
    cam = trial_info["camera"]

    if not cond["needs_video_degrade"]:
        # FPS-only : chercher un JSON extrait à ce FPS
        json_name = "{}_cam{}_{:.0f}fps.json".format(trial, cam, cond["fps"])
        json_path = json_dir / json_name

        if json_path.exists():
            with open(json_path) as f:
                return json.load(f)

        if skip_extraction:
            return None

        # Extraire la vidéo nativement (myogait extrait au FPS natif)
        # Pour un FPS inférieur, on pré-traite avec ffmpeg
        if cond["fps"] < 200:
            # Réduire le FPS via ffmpeg puis extraire
            tmp_video = json_dir / "{}_cam{}_{:.0f}fps.mp4".format(
                trial, cam, cond["fps"])
            cmd = ["ffmpeg", "-y", "-i", trial_info["video_path"],
                   "-r", str(cond["fps"]), "-an", str(tmp_video)]
            try:
                subprocess.run(cmd, capture_output=True, check=True, timeout=120)
            except Exception as e:
                print("    ffmpeg fps conversion failed: {}".format(e))
                return None
            data = extract_video(str(tmp_video), str(json_path))
            tmp_video.unlink(missing_ok=True)
            return data
        else:
            return extract_video(trial_info["video_path"], str(json_path))

    # Condition avec dégradation vidéo : chercher un JSON pré-extrait
    cond_dir = json_dir / cond["name"]
    json_name = "{}_cam{}.json".format(trial, cam)
    json_path = cond_dir / json_name

    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)

    if skip_extraction:
        return None

    # Dégrader la vidéo puis extraire
    cond_dir.mkdir(parents=True, exist_ok=True)
    degraded_video = cond_dir / "{}_cam{}.mp4".format(trial, cam)

    print("    Degrading video → {} ...".format(cond["name"]), end=" ", flush=True)
    if not degrade_video(trial_info["video_path"], degraded_video, cond):
        return None

    print("    Extracting degraded video ...", end=" ", flush=True)
    data = extract_video(str(degraded_video), str(json_path))
    degraded_video.unlink(missing_ok=True)
    return data


# ── Détection ────────────────────────────────────────────────────────────────

def process_and_detect(data, gt_info, butter_cutoff=4.0):
    """Normalise, calcule les angles, exécute tous les détecteurs.

    Returns (results_dict, t_video_start, t_video_end) or (None, None, None).
    The caller must filter GT events to [t_video_start, t_video_end] to avoid
    counting GT events outside the actual video coverage as false negatives.
    """
    import myogait

    # Trim aux bornes du valid range MoCap
    t_start = gt_info["t_start"]
    t_end = gt_info["t_end"]

    data["frames"] = [f for f in data["frames"]
                      if t_start <= f.get("time_s", 0) <= t_end]

    if len(data["frames"]) < 20:
        return None, None, None

    t_video_start = data["frames"][0]["time_s"]
    t_video_end = data["frames"][-1]["time_s"]

    # Normaliser + angles
    data = myogait.normalize(data, filters=["butterworth"],
                             butterworth_cutoff=butter_cutoff)
    data = myogait.compute_angles(data)

    # Exécuter chaque détecteur
    results = {}
    for method in DETECTORS:
        d = copy.deepcopy(data)
        try:
            d = myogait.detect_events(d, method=method)
            ev = d.get("events", {})
            hs = sorted(
                [e["time"] for e in ev.get("left_hs", [])] +
                [e["time"] for e in ev.get("right_hs", [])]
            )
            to = sorted(
                [e["time"] for e in ev.get("left_to", [])] +
                [e["time"] for e in ev.get("right_to", [])]
            )
            results[method] = {"hs": hs, "to": to}
        except Exception as e:
            results[method] = {"hs": [], "to": [], "error": str(e)}

    return results, t_video_start, t_video_end


# ── Évaluation ───────────────────────────────────────────────────────────────

def evaluate(det_times, gt_times, tol_s):
    """Match détections vs GT, retourne métriques."""
    if not gt_times:
        return {"tp": 0, "fp": len(det_times), "fn": 0,
                "precision": 0, "recall": 0, "f1": 0, "mae_ms": float("nan")}

    used_gt = set()
    tp_errors = []

    for dt in det_times:
        best_j, best_err = None, 9999
        for j, gt in enumerate(gt_times):
            if j not in used_gt and abs(dt - gt) < abs(best_err):
                best_j, best_err = j, dt - gt
        if best_j is not None and abs(best_err) <= tol_s:
            used_gt.add(best_j)
            tp_errors.append(best_err)

    tp = len(tp_errors)
    fp = len(det_times) - tp
    fn = len(gt_times) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    mae = np.mean(np.abs(tp_errors)) * 1000 if tp_errors else float("nan")

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mae_ms": round(mae, 1) if not np.isnan(mae) else "",
    }


# ── Résumé ───────────────────────────────────────────────────────────────────

def _micro_f1(tp_total, fp_total, fn_total):
    """Calcule le F1 micro à partir des TP/FP/FN agrégés."""
    prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0


def print_summary(rows):
    """Affiche le résumé par condition × détecteur (F1 macro/micro, MAE à 50ms)."""
    if not rows:
        return

    # Filtrer à 50ms
    rows_50 = [r for r in rows if r["tolerance_ms"] == 50]
    if not rows_50:
        return

    # Collecter les métriques par (condition, detector)
    stats = defaultdict(lambda: {
        "f1_hs": [], "f1_to": [], "mae_hs": [], "mae_to": [],
        "tp_hs": 0, "fp_hs": 0, "fn_hs": 0,
        "tp_to": 0, "fp_to": 0, "fn_to": 0,
    })

    for r in rows_50:
        key = (r["condition"], r["detector"])
        stats[key]["f1_hs"].append(r["f1_hs"])
        stats[key]["f1_to"].append(r["f1_to"])
        stats[key]["tp_hs"] += r["tp_hs"]
        stats[key]["fp_hs"] += r["fp_hs"]
        stats[key]["fn_hs"] += r["fn_hs"]
        stats[key]["tp_to"] += r["tp_to"]
        stats[key]["fp_to"] += r["fp_to"]
        stats[key]["fn_to"] += r["fn_to"]
        if r["mae_hs_ms"] != "":
            stats[key]["mae_hs"].append(float(r["mae_hs_ms"]))
        if r["mae_to_ms"] != "":
            stats[key]["mae_to"].append(float(r["mae_to_ms"]))

    conditions = []
    for c in DEGRADATION_CONDITIONS:
        if any(r["condition"] == c["name"] for r in rows_50):
            conditions.append(c)

    detectors = []
    for d in DETECTORS:
        dname = d.replace("gk_", "")
        if any(r["detector"] == dname for r in rows_50):
            detectors.append(dname)

    hdr = "{:<22s}".format("Condition")
    for d in detectors:
        hdr += " {:>9s}".format(d[:9])
    hdr += " {:>9s}".format("MOYENNE")

    # ── Helper pour afficher un tableau F1 ──
    def _print_f1_table(title, event_type, mode):
        """mode='macro' ou 'micro'"""
        print("\n" + "=" * 100)
        print("RÉSUMÉ — F1 {} {} à 50ms".format(mode, event_type.upper()))
        print("=" * 100)
        print(hdr)
        print("-" * len(hdr))

        for cond in conditions:
            line = "{:<22s}".format(cond["label"][:22])
            cond_vals = []
            for d in detectors:
                key = (cond["name"], d)
                if mode == "macro":
                    vals = stats[key]["f1_{}".format(event_type)]
                    if vals:
                        m = np.mean(vals)
                        cond_vals.append(m)
                        line += " {:>9.2f}".format(m)
                    else:
                        line += " {:>9s}".format("—")
                else:  # micro
                    tp = stats[key]["tp_{}".format(event_type)]
                    fp = stats[key]["fp_{}".format(event_type)]
                    fn = stats[key]["fn_{}".format(event_type)]
                    if tp + fp + fn > 0:
                        m = _micro_f1(tp, fp, fn)
                        cond_vals.append(m)
                        line += " {:>9.2f}".format(m)
                    else:
                        line += " {:>9s}".format("—")
            avg = np.mean(cond_vals) if cond_vals else float("nan")
            line += " {:>9.2f}".format(avg) if not np.isnan(avg) else " {:>9s}".format("—")
            print(line)

        # Moyenne par détecteur
        line = "{:<22s}".format("MOYENNE")
        for d in detectors:
            if mode == "macro":
                all_f1 = []
                for cond in conditions:
                    all_f1.extend(stats[(cond["name"], d)]["f1_{}".format(event_type)])
                m = np.mean(all_f1) if all_f1 else float("nan")
            else:
                tp = sum(stats[(c["name"], d)]["tp_{}".format(event_type)] for c in conditions)
                fp = sum(stats[(c["name"], d)]["fp_{}".format(event_type)] for c in conditions)
                fn = sum(stats[(c["name"], d)]["fn_{}".format(event_type)] for c in conditions)
                m = _micro_f1(tp, fp, fn) if (tp + fp + fn) > 0 else float("nan")
            line += " {:>9.2f}".format(m) if not np.isnan(m) else " {:>9s}".format("—")
        print(line)

    # ── 4 tableaux F1 : macro/micro × HS/TO ──
    _print_f1_table("F1 macro HS", "hs", "macro")
    _print_f1_table("F1 micro HS", "hs", "micro")
    _print_f1_table("F1 macro TO", "to", "macro")
    _print_f1_table("F1 micro TO", "to", "micro")

    # ── Tableau MAE HS ──
    print("\n" + "=" * 100)
    print("RÉSUMÉ — MAE moyen HS (ms) à 50ms")
    print("=" * 100)
    print(hdr)
    print("-" * len(hdr))

    for cond in conditions:
        line = "{:<22s}".format(cond["label"][:22])
        cond_maes = []
        for d in detectors:
            key = (cond["name"], d)
            vals = stats[key]["mae_hs"]
            if vals:
                m = np.mean(vals)
                cond_maes.append(m)
                line += " {:>8.1f}".format(m) + " "
            else:
                line += " {:>9s}".format("—")
        avg = np.mean(cond_maes) if cond_maes else float("nan")
        line += " {:>8.1f}".format(avg) if not np.isnan(avg) else " {:>9s}".format("—")
        print(line)

    # ── Tableau MAE TO ──
    print("\n" + "=" * 100)
    print("RÉSUMÉ — MAE moyen TO (ms) à 50ms")
    print("=" * 100)
    print(hdr)
    print("-" * len(hdr))

    for cond in conditions:
        line = "{:<22s}".format(cond["label"][:22])
        cond_maes = []
        for d in detectors:
            key = (cond["name"], d)
            vals = stats[key]["mae_to"]
            if vals:
                m = np.mean(vals)
                cond_maes.append(m)
                line += " {:>8.1f}".format(m) + " "
            else:
                line += " {:>9s}".format("—")
        avg = np.mean(cond_maes) if cond_maes else float("nan")
        line += " {:>8.1f}".format(avg) if not np.isnan(avg) else " {:>9s}".format("—")
        print(line)

    # ── Nombre de trials par condition ──
    print("\n--- Trials évalués par condition ---")
    for cond in conditions:
        n = len(set((r["subject"], r["trial"], r["camera"])
                    for r in rows_50 if r["condition"] == cond["name"]))
        print("  {:<22s}: {} trials".format(cond["label"], n))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark markerless gait event detection (BioCV)")
    parser.add_argument("--data-dir", required=True,
                        help="Chemin vers BATH-01258/")
    parser.add_argument("--output", required=True,
                        help="Fichier CSV de sortie")
    parser.add_argument("--json-dir", default=None,
                        help="Répertoire des JSONs myogait "
                             "(défaut: data-dir/../results/myogait_biocv/)")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Sujets à traiter (défaut: tous)")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Max trials WALK par sujet")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Conditions à tester (défaut: toutes). "
                             "Ex: lab_reference clinique_pro")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Ne pas extraire les vidéos, utiliser les JSONs "
                             "existants uniquement")
    parser.add_argument("--butter-cutoff", type=float, default=4.0,
                        help="Fréquence de coupure Butterworth (Hz)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    json_dir = (Path(args.json_dir) if args.json_dir
                else data_dir.parent / "results" / "myogait_biocv")
    json_dir.mkdir(parents=True, exist_ok=True)

    # Filtrer les conditions
    if args.conditions:
        conditions = [COND_BY_NAME[c] for c in args.conditions
                      if c in COND_BY_NAME]
    else:
        conditions = DEGRADATION_CONDITIONS

    # Découvrir les trials (caméras latérales uniquement)
    trials = find_trials(data_dir, args.subjects, args.max_trials)
    print("Found {} trial/camera combinations".format(len(trials)))
    print("Conditions: {}".format(
        ", ".join(c["name"] for c in conditions)))
    print("Detectors: {}".format(len(DETECTORS)))

    if not trials:
        print("No trials found!")
        sys.exit(1)

    # CSV header
    fieldnames = [
        "subject", "trial", "camera", "condition", "fps_video",
        "detector", "tolerance_ms",
        "n_gt_hs", "n_det_hs", "tp_hs", "fp_hs", "fn_hs",
        "precision_hs", "recall_hs", "f1_hs", "mae_hs_ms",
        "n_gt_to", "n_det_to", "tp_to", "fp_to", "fn_to",
        "precision_to", "recall_to", "f1_to", "mae_to_ms",
    ]

    rows = []
    n_ok = 0
    n_fail = 0
    n_cond_skip = 0

    # Ouvrir le CSV dès le début et écrire au fur et à mesure
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(output_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_file.flush()

    for i, trial_info in enumerate(trials):
        subj = trial_info["subject"]
        trial = trial_info["trial"]
        cam = trial_info["camera"]

        print("\n[{}/{}] {} cam{}".format(i + 1, len(trials), trial, cam))

        # 1. GT events
        try:
            gt = get_gt_events(trial_info["c3d_path"])
        except Exception as e:
            print("  GT extraction failed: {}".format(e))
            n_fail += 1
            continue

        if len(gt["hs"]) == 0 and len(gt["to"]) == 0:
            print("  No GT events, skipping")
            n_fail += 1
            continue

        print("  GT: {} HS, {} TO, valid [{:.2f}, {:.2f}]s".format(
            len(gt["hs"]), len(gt["to"]), gt["t_start"], gt["t_end"]))

        # 2. Pour chaque condition de dégradation
        trial_ok = False

        for cond in conditions:
            data = get_data_for_condition(
                trial_info, cond, json_dir, args.skip_extraction)

            if data is None:
                n_cond_skip += 1
                continue

            fps_video = cond["fps"]

            # Trim + normalise + détection
            try:
                det_results, t_vs, t_ve = process_and_detect(
                    copy.deepcopy(data), gt, args.butter_cutoff)
            except Exception as e:
                print("  [{}] Detection failed: {}".format(
                    cond["name"], e))
                traceback.print_exc()
                n_cond_skip += 1
                continue

            if det_results is None:
                print("  [{}] Too few frames in valid range".format(
                    cond["name"]))
                n_cond_skip += 1
                continue

            # Filter GT to actual video coverage (avoid FN for events
            # outside the video temporal extent)
            margin = 0.010  # 10ms margin
            gt_hs_vid = [t for t in gt["hs"]
                         if t_vs - margin <= t <= t_ve + margin]
            gt_to_vid = [t for t in gt["to"]
                         if t_vs - margin <= t <= t_ve + margin]

            # Évaluer chaque détecteur à chaque tolérance
            for method, det in det_results.items():
                det_name = method.replace("gk_", "")
                n_det_hs = len(det["hs"])
                n_det_to = len(det["to"])

                for tol_ms in TOLERANCES_MS:
                    tol_s = tol_ms / 1000.0
                    hs_m = evaluate(det["hs"], gt_hs_vid, tol_s)
                    to_m = evaluate(det["to"], gt_to_vid, tol_s)

                    row = {
                        "subject": subj,
                        "trial": trial,
                        "camera": "cam{}".format(cam),
                        "condition": cond["name"],
                        "fps_video": fps_video,
                        "detector": det_name,
                        "tolerance_ms": tol_ms,
                        "n_gt_hs": len(gt_hs_vid),
                        "n_det_hs": n_det_hs,
                        "tp_hs": hs_m["tp"],
                        "fp_hs": hs_m["fp"],
                        "fn_hs": hs_m["fn"],
                        "precision_hs": hs_m["precision"],
                        "recall_hs": hs_m["recall"],
                        "f1_hs": hs_m["f1"],
                        "mae_hs_ms": hs_m["mae_ms"],
                        "n_gt_to": len(gt_to_vid),
                        "n_det_to": n_det_to,
                        "tp_to": to_m["tp"],
                        "fp_to": to_m["fp"],
                        "fn_to": to_m["fn"],
                        "precision_to": to_m["precision"],
                        "recall_to": to_m["recall"],
                        "f1_to": to_m["f1"],
                        "mae_to_ms": to_m["mae_ms"],
                    }
                    rows.append(row)
                    csv_writer.writerow(row)

            csv_file.flush()
            trial_ok = True

            # Résumé rapide pour ce trial × condition (50ms)
            summary_parts = []
            for method in ["gk_bike", "gk_zeni", "gk_intellevent"]:
                if method in det_results:
                    det = det_results[method]
                    hs_50 = evaluate(det["hs"], gt_hs_vid, 0.050)
                    to_50 = evaluate(det["to"], gt_to_vid, 0.050)
                    dn = method.replace("gk_", "")
                    mae_hs_str = "{:.0f}".format(hs_50["mae_ms"]) if hs_50["mae_ms"] != "" else "—"
                    mae_to_str = "{:.0f}".format(to_50["mae_ms"]) if to_50["mae_ms"] != "" else "—"
                    summary_parts.append(
                        "{}: HS F1={:.2f} MAE={}ms TO F1={:.2f} MAE={}ms".format(
                            dn, hs_50["f1"], mae_hs_str, to_50["f1"], mae_to_str))
            print("  [{}] {}".format(
                cond["name"], " | ".join(summary_parts)))

        if trial_ok:
            n_ok += 1
        else:
            n_fail += 1

    csv_file.close()

    print("\n" + "=" * 100)
    print("DONE: {} trials OK, {} failed, {} condition-skips".format(
        n_ok, n_fail, n_cond_skip))
    print("Results written to: {}".format(output_path))
    print("{} rows".format(len(rows)))

    # 5. Résumé
    print_summary(rows)


if __name__ == "__main__":
    main()
