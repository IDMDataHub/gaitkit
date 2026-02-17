# Changelog

## Unreleased

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
