# gaitkit (Python)

Python package for gait event detection from motion-capture data.

## Install

Current status: PyPI release available (latest stable on PyPI).

Install from local source:

```bash
python -m pip install -e ./python
python -m pip install -e "./python[all]"   # optional extras
```

Once published:

```bash
pip install gaitkit
```

Optional extras:

```bash
pip install "gaitkit[all]"   # onnx + deep + viz
```

## Quick Start

```python
import gaitkit

trial = gaitkit.load_example("healthy")
result = gaitkit.detect(trial, method="bike")
print(result.summary())

# Optional: combine C3D markers with an external angle file
result2 = gaitkit.detect("trial_07.c3d", method="bike", angles="res_angles_t.mat")
```

DeepEvent weights are downloaded automatically on first DeepEvent use and
cached in `~/.cache/gaitkit/`.

## Project

- Repository: https://github.com/IDMDataHub/gaitkit
- Issue tracker: https://github.com/IDMDataHub/gaitkit/issues
- Reproducibility: https://github.com/IDMDataHub/gaitkit/blob/master/REPRODUCIBILITY.md

## Development testing

```bash
python -m pip install -e ./python
python -m unittest -v
```

## Build distributions

```bash
python -m pip install build
cd python
python -m build
```
