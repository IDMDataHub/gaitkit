# MATLAB wrapper (gaitkit)

MATLAB wrapper around `py.gaitkit`.

## Usage

```matlab
addpath('gaitkit/matlab')
pyenv("Version","/path/to/python")  % optional, to select interpreter
methods = gaitkit.listMethods();
res = gaitkit.detect('bayesian_bis', frames, 100, struct('position','mm','angles','deg'));
```

`frames` must be a struct array compatible with
`gaitkit.detect_events_structured` frame fields.

If `method` is omitted or empty, `bike` is used by default.
`units` must be a struct with:
- `position`: `'mm'` or `'m'`
- `angles`: `'deg'` or `'rad'`

## Install backend from MATLAB

```matlab
gaitkit.installPythonBackend();
gaitkit.installPythonBackend("gaitkit[all]"); % optional extras
```

If installation fails, the wrapper now reports full `pip` output to help
diagnose environment/network issues.
