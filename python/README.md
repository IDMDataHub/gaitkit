# gaitkit (Python)

Python package for gait event detection from motion-capture data.

## Install

```bash
pip install gaitkit
```

Optional extras:

```bash
pip install "gaitkit[all]"   # onnx + deep + viz + c3d helpers
```

## Quick Start

```python
import gaitkit

trial = gaitkit.load_example("healthy")
result = gaitkit.detect(trial, method="bike")
print(result.summary())
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
