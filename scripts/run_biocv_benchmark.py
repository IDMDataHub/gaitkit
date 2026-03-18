#!/usr/bin/env python3
"""
BioCV Benchmark — Run all gaitkit detectors on the BioCV dataset.

Produces output in the exact same format as full_benchmark.csv used by
the BIKE article (with population, tolerance_ms, hs_tp/fp/fn, event_source,
coverage_adjusted columns).

Usage (on server):
    source ~/gait_venv/bin/activate
    python ~/gaitkit/scripts/run_biocv_benchmark.py \
        --data-dir ~/gait_benchmark_project/data/BATH-01258 \
        --output   ~/gait_benchmark_project/results/benchmark_biocv.csv

    # Append to existing full_benchmark.csv:
    python ~/gaitkit/scripts/run_biocv_benchmark.py \
        --data-dir ~/gait_benchmark_project/data/BATH-01258 \
        --append-to ~/bike_article/data/full_benchmark.csv

Dependencies: gaitkit >= 1.4.0
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import gaitkit
from gaitkit.extractors.biocv_extractor import BioCVExtractor


# ---------------------------------------------------------------------------
# Article-format columns
# ---------------------------------------------------------------------------
ARTICLE_COLUMNS = [
    "dataset", "population", "source_file", "subject_id", "trial_id",
    "condition", "n_frames", "fps", "duration_s",
    "n_gt_hs", "n_gt_to", "n_gt_hs_left", "n_gt_hs_right",
    "n_gt_to_left", "n_gt_to_right",
    "detector_name", "n_detected_hs", "n_detected_to",
    "hs_precision", "hs_recall", "hs_f1", "hs_mae_ms",
    "hs_tp", "hs_fp", "hs_fn",
    "to_precision", "to_recall", "to_f1", "to_mae_ms",
    "to_tp", "to_fp", "to_fn",
    "processing_time_ms", "tolerance_ms",
    "event_source", "coverage_adjusted",
]

# Detectors used in the article.  gaitkit method names → article CSV names.
GAITKIT_TO_ARTICLE = {
    "bike":           "bayesian_bis",
    "zeni":           "zeni",
    "oconnor":        "oconnor",
    "hreljac":        "hreljac",
    "mickelborough":  "mickelborough",
    "ghoussayni":     "ghoussayni",
    "vancanneyt":     "vancanneyt",
    "dgei":           "dgei",
    "intellevent":    "intellevent",
    "deepevent":      "deepevent",
}

ARTICLE_TOLERANCES_MS = [25, 50, 75]


# ---------------------------------------------------------------------------
# Event matching (same greedy algorithm as article)
# ---------------------------------------------------------------------------
def match_events(
    detected: List[int],
    ground_truth: List[int],
    tolerance_frames: int,
) -> Tuple[float, float, float, float, int, int, int]:
    """Returns (prec, rec, f1, mae_frames, tp, fp, fn)."""
    if not ground_truth and not detected:
        return (1.0, 1.0, 1.0, 0.0, 0, 0, 0)
    if not ground_truth:
        return (0.0, 0.0, 0.0, 0.0, 0, len(detected), 0)
    if not detected:
        return (0.0, 0.0, 0.0, 0.0, 0, 0, len(ground_truth))

    gt_matched = [False] * len(ground_truth)
    det_matched = [False] * len(detected)
    errors = []

    for i, det in enumerate(detected):
        best_j, best_dist = -1, float("inf")
        for j, gt in enumerate(ground_truth):
            if gt_matched[j]:
                continue
            dist = abs(det - gt)
            if dist <= tolerance_frames and dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j >= 0:
            gt_matched[best_j] = True
            det_matched[i] = True
            errors.append(best_dist)

    tp = sum(det_matched)
    fp = len(detected) - tp
    fn = len(ground_truth) - sum(gt_matched)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    mae_frames = float(np.mean(errors)) if errors else 0.0
    return (prec, rec, f1, mae_frames, tp, fp, fn)


# ---------------------------------------------------------------------------
# Run one gaitkit method on angle_frames
# ---------------------------------------------------------------------------
def run_gaitkit_detect(method: str, angle_frames, fps: float, frame_offset: int = 0):
    """Run gaitkit.detect() and return (hs_frame_list, to_frame_list, time_ms).

    Detectors return 0-based list indices.  When the extractor trims to a
    valid frame range, ``frame_offset`` (= valid_frame_range[0]) is added to
    convert detected indices back to absolute frame numbers matching GT.
    """
    t0 = time.time()
    try:
        result = gaitkit.detect(angle_frames, fps=fps, method=method)
    except Exception as exc:
        print(f"      {method} error: {exc}")
        return [], [], 0.0
    elapsed = (time.time() - t0) * 1000

    events = result.events
    hs_frames = sorted((events.loc[events["event_type"] == "HS", "frame"].astype(int) + frame_offset).tolist())
    to_frames = sorted((events.loc[events["event_type"] == "TO", "frame"].astype(int) + frame_offset).tolist())
    return hs_frames, to_frames, elapsed


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_biocv_benchmark(data_dir: str, methods: dict, tolerances: List[int]):
    """Run benchmark. methods = {gaitkit_name: article_name}."""
    ext = BioCVExtractor(data_dir)
    files = ext.list_files()
    print(f"BioCV: found {len(files)} trials")

    if not files:
        print("No files found! Check that data is extracted.")
        return pd.DataFrame(columns=ARTICLE_COLUMNS)

    rows = []
    for fi, fpath in enumerate(files):
        trial_name = fpath.parent.name
        print(f"  [{fi+1}/{len(files)}] {trial_name} ...", flush=True)

        try:
            extraction = ext.extract_file(fpath)
        except Exception as exc:
            print(f"    extraction error: {exc}")
            continue

        gt = extraction.ground_truth
        if not gt.has_hs and not gt.has_to:
            print("    skip (no GT events)")
            continue

        fps = extraction.fps
        gt_hs = sorted(gt.hs_frames.get("left", []) + gt.hs_frames.get("right", []))
        gt_to = sorted(gt.to_frames.get("left", []) + gt.to_frames.get("right", []))

        # Frame offset: detectors return 0-based indices into the
        # (possibly trimmed) angle_frames list.  Add offset to convert
        # back to absolute frame numbers matching GT events.
        frame_offset = gt.valid_frame_range[0] if gt.valid_frame_range else 0

        for gk_method, article_name in methods.items():
            det_hs, det_to, proc_ms = run_gaitkit_detect(
                gk_method, extraction.angle_frames, fps, frame_offset
            )
            print(f"    {article_name}: {len(det_hs)} HS, {len(det_to)} TO ({proc_ms:.0f}ms)")

            for tol_ms in tolerances:
                tol_frames = int(tol_ms / 1000.0 * fps)

                hs_p, hs_r, hs_f1, hs_mae_f, hs_tp, hs_fp, hs_fn = \
                    match_events(det_hs, gt_hs, tol_frames)
                to_p, to_r, to_f1, to_mae_f, to_tp, to_fp, to_fn = \
                    match_events(det_to, gt_to, tol_frames)

                hs_mae_ms = (hs_mae_f / fps * 1000) if hs_mae_f > 0 else 0.0
                to_mae_ms = (to_mae_f / fps * 1000) if to_mae_f > 0 else 0.0

                rows.append({
                    "dataset": "biocv",
                    "population": "healthy",
                    "source_file": fpath.name,
                    "subject_id": extraction.subject_id,
                    "trial_id": extraction.trial_id,
                    "condition": extraction.condition,
                    "n_frames": extraction.n_frames,
                    "fps": fps,
                    "duration_s": round(extraction.duration_s, 3),
                    "n_gt_hs": len(gt_hs),
                    "n_gt_to": len(gt_to),
                    "n_gt_hs_left": len(gt.hs_frames.get("left", [])),
                    "n_gt_hs_right": len(gt.hs_frames.get("right", [])),
                    "n_gt_to_left": len(gt.to_frames.get("left", [])),
                    "n_gt_to_right": len(gt.to_frames.get("right", [])),
                    "detector_name": article_name,
                    "n_detected_hs": len(det_hs),
                    "n_detected_to": len(det_to),
                    "hs_precision": round(hs_p, 6),
                    "hs_recall": round(hs_r, 6),
                    "hs_f1": round(hs_f1, 6),
                    "hs_mae_ms": round(hs_mae_ms, 6),
                    "hs_tp": hs_tp, "hs_fp": hs_fp, "hs_fn": hs_fn,
                    "to_precision": round(to_p, 6),
                    "to_recall": round(to_r, 6),
                    "to_f1": round(to_f1, 6),
                    "to_mae_ms": round(to_mae_ms, 6),
                    "to_tp": to_tp, "to_fp": to_fp, "to_fn": to_fn,
                    "processing_time_ms": round(proc_ms, 6),
                    "tolerance_ms": tol_ms,
                    "event_source": gt.event_source,
                    "coverage_adjusted": gt.valid_frame_range is not None,
                })

    df = pd.DataFrame(rows, columns=ARTICLE_COLUMNS)
    n_det = len(methods)
    n_tol = len(tolerances)
    print(f"\nTotal: {len(df)} rows ({len(files)} trials x {n_det} detectors x {n_tol} tolerances)")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run BIKE benchmark on BioCV dataset")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to BioCV data (e.g. ~/gait_benchmark_project/data/BATH-01258)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")
    parser.add_argument("--append-to", type=str, default=None,
                        help="Append results to existing full_benchmark.csv")
    parser.add_argument("--detectors", type=str, nargs="+", default=None,
                        help="gaitkit method names to run (default: all 9 article detectors)")
    parser.add_argument("--tolerances", type=int, nargs="+", default=None,
                        help=f"Tolerances in ms (default: {ARTICLE_TOLERANCES_MS})")
    args = parser.parse_args()

    # Build method dict
    available = gaitkit.list_methods()
    if args.detectors:
        methods = {}
        for d in args.detectors:
            if d in GAITKIT_TO_ARTICLE:
                methods[d] = GAITKIT_TO_ARTICLE[d]
            elif d in available:
                methods[d] = d
            else:
                print(f"WARNING: '{d}' not in gaitkit methods {available}, skipping")
    else:
        methods = {k: v for k, v in GAITKIT_TO_ARTICLE.items() if k in available}

    tolerances = args.tolerances or ARTICLE_TOLERANCES_MS

    print("=" * 70)
    print("BIOCV BENCHMARK (gaitkit detectors)")
    print("=" * 70)
    print(f"Data:       {args.data_dir}")
    print(f"Detectors:  {list(methods.values())}")
    print(f"Tolerances: {tolerances} ms")
    print(f"gaitkit:    {gaitkit.__version__}")
    print("=" * 70)

    df = run_biocv_benchmark(args.data_dir, methods, tolerances)

    if df.empty:
        print("No results produced.")
        return

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("results") / f"benchmark_biocv_{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df)} rows)")

    # Append
    if args.append_to:
        append_path = Path(args.append_to)
        if append_path.exists():
            existing = pd.read_csv(append_path)
            existing = existing[existing["dataset"] != "biocv"]
            combined = pd.concat([existing, df], ignore_index=True)
            combined.to_csv(append_path, index=False)
            print(f"Appended to {append_path} ({len(existing)} + {len(df)} = {len(combined)})")
        else:
            df.to_csv(append_path, index=False)
            print(f"Created {append_path} ({len(df)} rows)")

    # Summary
    print("\n" + "=" * 70)
    print("QUICK SUMMARY (tolerance=50ms)")
    print("=" * 70)
    df50 = df[df["tolerance_ms"] == 50]
    if not df50.empty:
        for event in ["hs", "to"]:
            col = f"{event}_f1"
            summary = df50.groupby("detector_name")[col].agg(["mean", "std", "count"])
            summary = summary.sort_values("mean", ascending=False)
            summary.columns = ["mean_F1", "std_F1", "n"]
            print(f"\n{event.upper()} F1:")
            print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
