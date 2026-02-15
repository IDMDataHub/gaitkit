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

**Python** (requires >= 3.9):

```bash
pip install gaitkit            # core (NumPy + SciPy)
pip install gaitkit[all]       # with ONNX, visualization, C3D support
```

DeepEvent model weights (`DeepEventWeight.h5`) are downloaded automatically
on first DeepEvent use and cached in `~/.cache/gaitkit/`.

**R** (requires R >= 4.1):

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
```

### R

```r
library(gaitkit)

trial <- gk_load_example("healthy")
events <- gk_detect(trial, method = "bike")
events$left_hs
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
| 1 | **BIKE** | Bayesian + kinematic | Fer et al. (2026) |
| 2 | Zeni | Kinematic (AP coord) | Zeni et al. (2008) |
| 3 | O'Connor | Kinematic (velocity) | O'Connor et al. (2007) |
| 4 | Hreljac | Kinematic (acceleration) | Hreljac & Marshall (2000) |
| 5 | Mickelborough | Kinematic (velocity) | Mickelborough et al. (2000) |
| 6 | Ghoussayni | Kinematic (velocity) | Ghoussayni et al. (2004) |
| 7 | DGEI | Kinematic (energy index) | Desailly et al. (2009) |
| 8 | Vancanneyt | Kinematic (wavelet) | Vancanneyt et al. (2022) |
| 9 | IntellEvent | Deep learning (BiLSTM) | Lempereur et al. (2020) |
| 10 | DeepEvent | Deep learning (CNN) | Kidzinski et al. (2019) |

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

If you use gaitkit in your research, please cite:

```bibtex
@article{fer2026gaitkit,
  title   = {gaitkit: A universal toolkit for gait event detection
             from motion capture data},
  author  = {Fer, Fr{\'e}d{\'e}ric},
  journal = {Journal of NeuroEngineering and Rehabilitation},
  year    = {2026}
}
```

---

## License

MIT -- see [LICENSE](LICENSE).

## Project Roadmap

Publication/community readiness tasks are tracked in [TODO.md](TODO.md).
