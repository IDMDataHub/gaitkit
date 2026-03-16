# Changelog

## 1.4.7 (2026-03-16)

### Fixed
- BioCV extractor: compute valid marker range (`valid_frame_range`) from the
  intersection of 10 critical markers being non-zero.  Visual3D joint centres
  are zero-padded outside the capture volume; this fix trims angle frames and
  GT events to the valid range, eliminating artefacts for event detectors.
- All detectors (BIKE, DGEI, Vancanneyt, Zeni): preserve original MoCap
  defaults at ≥100 fps.  FPS-adaptive parameters only apply below 100 fps,
  ensuring zero regression on marker-based benchmarks.

## 1.4.6 (2026-03-15)

### Improved
- BIKE: all internal windows (SavGol, p_signal sigma, half_win) are now
  defined in milliseconds and converted to frames, making the detector
  FPS-independent.  Previously hard-coded frame counts caused over-smoothing
  at low frame rates (e.g. 30 fps markerless video).
  At MoCap rates (≥100 fps) original defaults are preserved — zero regression.
- BIKE: sub-frame interpolation on the diff zero-crossing refines event
  timing beyond integer-frame precision.
- BIKE: SavGol bias compensation at low FPS (<100 fps) corrects the
  systematic forward shift introduced by polynomial smoothing on coarsely
  sampled trajectories.
- DGEI, Vancanneyt, Zeni: FPS-adaptive smoothing/windowing at low frame
  rates (<100 fps) for markerless video; original MoCap defaults preserved.

## 1.4.5 (2026-03-15)

### Fixed
- Enforce global HS/TO alternation after boundary and gap event insertion.
  `_boundary_events` and `_gap_events` could produce consecutive same-type
  events (e.g. HS, HS) on noisy input signals; new `_enforce_alternation`
  post-processing removes the lower-probability duplicate.

## Unreleased

## 1.2.4 (2026-02-20)

### Added
- BikeGait compatibility now supports MyoGait JSON payloads as input.
- Added MyoGait-compatible event export format (`formats=("myogait",)`).

### Improved
- R `gk_detect()` now forwards C3D angle options (`angles`, `angles_align`,
  `require_core_markers`) to the Python backend.
- Added regression tests for MyoGait payload handling and R argument forwarding.

## 1.2.3 (2026-02-17)

### Fixed
- Improved C3D opening diagnostics for Windows/OneDrive path issues.
- Added strict optional core-marker validation for marker-only C3D workflows.
- Added label normalization for common vendor suffixes (e.g. `LHEE:1`).

## 1.2.2 (2026-02-16)

### Fixed
- Compute proxy angles automatically from markers when C3D model angles are missing.
- Improve IMY proxy-angle fidelity (hip/knee in X-Z, ankle in Y-Z, baseline normalization).
- Align verification helper with the exact angles used by the runtime C3D loader.
- Robustly align shorter external angle series to gait events when needed.

### Improved
- Hardened Python/R/MATLAB input validation across wrappers and detectors.
- Expanded non-regression and unit test coverage (Python + R bridge).
- Improved optional-dependency portability for extractor imports.
- Clarified install, testing, and reproducibility documentation.
- Strengthened evaluation helper input validation (metrics/matching/statistics).
- Removed silent fallbacks (`except/pass`) in extractors and detector wrappers.
- Simplified native-wrapper import logic to expected `ImportError` paths only.
- Added explicit Python/R packaging verification commands and cleaner build ignores.

## 0.1.0 (2026-02-15)

Initial release.

### Features
- 10 gait event detection methods: BIKE (default), Zeni, OConnor, Hreljac,
  Mickelborough, Ghoussayni, Vancanneyt, DGEI, IntellEvent, DeepEvent
- C-accelerated native backends for 8 methods (up to 13x speedup)
- Ensemble voting mode with temporal clustering and confidence scores
- GaitResult class with events/cycles DataFrames, CSV/JSON export
- Visualisation: timeline plots, butterfly gait cycles, ensemble confidence
- 4 bundled example datasets: healthy, Parkinson, stroke, force-plate
- C3D file I/O via ezc3d
- R package (gk_detect, gk_methods, S3 print/summary/plot)
- MATLAB wrapper (+gaitkit namespace)
- Bundled model weights: IntellEvent ONNX (12 MB), DeepEvent H5 (414 MB via LFS)
