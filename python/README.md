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

## Proprietary JSON I/O (MyoGait-like)

The compatibility API accepts a proprietary JSON payload with
`angles.frames` and can export back a MyoGait-compatible `events` JSON.

```python
import gaitkit

out = gaitkit.detect_events_structured("bike", "myogait_output_no_events.json", fps=100.0)
paths = gaitkit.export_detection(out, "outputs/trial_07", formats=("json", "myogait"))
print(paths)
```

Input fields recognized per frame:
- `frame_idx`
- `hip_L`, `knee_L`, `ankle_L`
- `hip_R`, `knee_R`, `ankle_R`
- `pelvis_tilt`, `trunk_angle`
- optional `landmark_positions`

MyoGait-compatible output file (`*_myogait_events.json`) contains:
- `events.method`
- `events.fps`
- `events.left_hs`, `events.right_hs`, `events.left_to`, `events.right_to`
  as arrays of `{frame, time, confidence}`.

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
