# MATLAB wrapper (gaitkit)

MATLAB wrapper around `py.gaitkit`.

## Usage

```matlab
addpath('gaitkit/matlab')
methods = gaitkit.listMethods();
res = gaitkit.detect('bayesian_bis', frames, 100, struct('position','mm','angles','deg'));
```

`frames` must be a struct array compatible with
`gaitkit.detect_events_structured` frame fields.

## Install backend from MATLAB

```matlab
gaitkit.installPythonBackend();
```
