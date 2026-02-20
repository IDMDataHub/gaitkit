# MATLAB wrapper (gaitkit)

MATLAB wrapper around `py.gaitkit`.

## Usage

```matlab
addpath('gaitkit/matlab')
pyenv("Version","/path/to/python")  % optional, to select interpreter
methods = gaitkit.listMethods();
res = gaitkit.detect('bike', frames, 100, struct('position','mm','angles','deg'));
```

`frames` must be a struct array compatible with
`gaitkit.detect_events_structured` frame fields.
You can also pass a path to a proprietary MyoGait-like JSON payload
(`angles.frames`), for example: `gaitkit.detect('bike', "myogait_output_no_events.json", 100)`.

If `method` is omitted or empty, `bike` is used by default.
`units` must be a struct with:
- `position`: `'mm'` or `'m'`
- `angles`: `'deg'` or `'rad'`

Result format is JSON-like with:
- `meta` (`detector`, `fps_hz`, `n_frames`, `available_methods`)
- `heel_strikes` and `toe_offs` (entries: `frame_index`, `time_s`, `side`, `confidence`)
- `cycles` (`start_frame`, `toe_off_frame`, `end_frame`, `duration`, `stance_percentage`)

## Install backend from MATLAB

```matlab
gaitkit.installPythonBackend();
gaitkit.installPythonBackend("gaitkit[all]"); % optional extras
```

If installation fails, the wrapper now reports full `pip` output to help
diagnose environment/network issues.
