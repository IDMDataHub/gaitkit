# gaitkit

**Universal gait event detection toolkit for motion capture data.**

Ten validated detection methods -- including the Bayesian BIKE detector -- with
C-accelerated backends, accessible from Python, R, and MATLAB.

<!--
[![PyPI](https://img.shields.io/pypi/v/gaitkit)](https://pypi.org/project/gaitkit/)
[![CRAN](https://img.shields.io/cran/v/gaitkit)](https://cran.r-project.org/package=gaitkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
-->

---

## Installation

Current status: PyPI release available (`gaitkit==1.2.3`), release `1.2.4` in progress, CRAN submission in progress.

**Install from local source (recommended for now)**

**Python** (requires >= 3.10):

```bash
python -m pip install -e ./python
python -m pip install -e "./python[all]"   # optional extras
```

**R** (requires R >= 4.1):

```r
install.packages(c("reticulate", "jsonlite", "devtools"))
devtools::install_local("r")
```

**MATLAB** (requires R2021b+):

```matlab
addpath("path/to/gaitkit/matlab")
gaitkit.installPythonBackend()   % one-time setup
```

**Once published**

**Python**:

```bash
pip install gaitkit            # core (NumPy + SciPy)
pip install gaitkit[all]       # with ONNX + deep-learning + visualization extras
```

DeepEvent model weights (`DeepEventWeight.h5`) are downloaded automatically
on first DeepEvent use and cached in `~/.cache/gaitkit/`.

**R**:

```r
# install.packages("remotes")
remotes::install_github("IDMDataHub/gaitkit", subdir = "r")
```

The R package calls the Python backend through `reticulate`.
Make sure `gaitkit` is installed in the Python environment used by R, or run:

```r
reticulate::use_python("/path/to/python", required = TRUE)  # optional but recommended
library(gaitkit)
gk_install_python()
gk_install_python(package = "gaitkit[all]")  # optional extras
```

**MATLAB** (requires R2021b+):

```matlab
addpath("path/to/gaitkit/matlab")
gaitkit.installPythonBackend()   % one-time setup
```

---

## Quick start

### Python

```python
import gaitkit

# Load a C3D trial
trial = gaitkit.load_c3d("walk_01.c3d")

# Detect gait events (defaults to BIKE method)
events = gaitkit.detect(trial, method="bike")

print(events.left_hs)   # list of {frame, time, side, confidence}
print(events.right_to)

# Optional: merge external angles (MAT/CSV/JSON) with C3D markers
events_with_angles = gaitkit.detect("walk_01.c3d", angles="res_angles_t.mat", method="bike")

# Optional: enforce a strict core marker set for marker-only C3D files
trial_strict = gaitkit.load_c3d("walk_01.c3d", require_core_markers=True)
```

Marker-only C3D processing is most robust when these canonical landmarks are
available on both sides: `heel`, `toe`, `ankle`, `knee`, `hip`.
If your labels differ, pass a custom `marker_map`.

### Proprietary JSON compatibility (MyoGait-like)

`gaitkit.detect_events_structured(...)` also accepts a proprietary payload with
`angles.frames` (dictionary or `.json` file path). Example:

```python
payload = gaitkit.detect_events_structured(
    "bike",
    "myogait_output_no_events.json",  # or a loaded dict payload
    fps=100.0
)
```

The helper `gaitkit.export_detection(...)` can export standard formats and a
MyoGait-compatible events JSON:

```python
paths = gaitkit.export_detection(payload, "out/trial_07", formats=("json", "myogait"))
print(paths["myogait"])  # out/trial_07_myogait_events.json
```

Expected proprietary JSON shape (input):

```json
{
  "meta": { "fps": 100.0 },
  "angles": {
    "frames": [
      {
        "frame_idx": 0,
        "hip_L": 10.2, "knee_L": 20.1, "ankle_L": -2.3,
        "hip_R": 11.0, "knee_R": 18.5, "ankle_R": -1.8,
        "pelvis_tilt": 3.1,
        "trunk_angle": 5.4,
        "landmark_positions": { "left_heel": [0.0, 0.0, 0.0] }
      }
    ]
  }
}
```

MyoGait-compatible output (`formats=("myogait",)`):

```json
{
  "events": {
    "method": "bike",
    "fps": 100.0,
    "left_hs": [{ "frame": 123, "time": 1.23, "confidence": 1.0 }],
    "right_hs": [],
    "left_to": [],
    "right_to": []
  }
}
```

### R

```r
library(gaitkit)

trial <- gk_load_example("healthy")
events <- gk_detect(trial, method = "bike")
events$left_hs

# Proprietary JSON input is also supported
events2 <- gk_detect("myogait_output_no_events.json", method = "bike")

# Export to MyoGait-compatible events JSON
gk_export_detection(events2, tempfile("gaitkit_out"), formats = c("json", "myogait"))
```

---

## Features

- **10 detection methods** spanning kinematic, kinetic, and deep-learning approaches
- **C-accelerated core** for real-time-capable throughput
- **Multi-language**: first-class Python, R, and MATLAB interfaces
- **C3D native I/O** via ezc3d -- no manual marker extraction
- **Built-in evaluation**: matching, F1, timing error metrics
- **Plug & Play architecture**: add a new detector with a single function

---

## Supported methods

| # | Method | Type | Reference |
|---|--------|------|-----------|
| 1 | **BIKE** | Bayesian + kinematic | This software (gaitkit) |
| 2 | Zeni | Kinematic (AP coord) | Zeni et al. (2008) |
| 3 | O'Connor | Kinematic (velocity) | O'Connor et al. (2007) |
| 4 | Hreljac | Kinematic (acceleration) | Hreljac & Marshall (2000) |
| 5 | Mickelborough | Kinematic (velocity) | Mickelborough et al. (2000) |
| 6 | Ghoussayni | Kinematic (velocity) | Ghoussayni et al. (2004) |
| 7 | DGEI | Kinematic (energy index) | Desailly et al. (2009) |
| 8 | Vancanneyt | Kinematic (wavelet) | Vancanneyt et al. (2022) |
| 9 | IntellEvent | Deep learning (BiLSTM) | Horsak & Kranzl (2023) |
| 10 | DeepEvent | Deep learning (BiLSTM) | Lempereur et al. (2020) |

---

## Example datasets

| Dataset | N | Population | Condition | Freq (Hz) |
|---------|---|------------|-----------|-----------|
| Fukuchi | 42 | Healthy | Treadmill + overground | 200 |
| Schreiber | 50 | Healthy | 5 speeds | 100--250 |
| Figshare PD | 26 | Parkinson | ON / OFF medication | 100 |
| Van Criekinge | 188 | Healthy + Stroke | Lifespan + post-AVC | 100 |
| Hood | 13 | Transfemoral amputee | Treadmill | 200 |

---

## Citation

If you use gaitkit in your research, please cite the software release:

```bibtex
@software{fer2026gaitkit,
  author  = {Fer, Fr{\'e}d{\'e}ric},
  title   = {gaitkit},
  version = {1.2.4},
  year    = {2026},
  doi     = {10.5281/zenodo.18653110},
  url     = {https://doi.org/10.5281/zenodo.18653110}
}
```

---

## License

MIT -- see [LICENSE](LICENSE).
Third-party model assets and their licenses are documented in
[THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md).

## Project Roadmap

Publication/community readiness tasks are tracked in [TODO.md](TODO.md).
Release and citation policy is documented in [RELEASE.md](RELEASE.md).
Method and dataset references are centralized in [REFERENCES.md](REFERENCES.md).

## Community

- Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Reproducibility checklist: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- Python and R packaging validation commands are documented in the reproducibility checklist.
- Detector registry smoke tests are included in reproducibility checks.
- Evaluation guardrail smoke tests are included in reproducibility checks.
