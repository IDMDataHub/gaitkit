# MATLAB wrapper (gaitkit)

MATLAB wrapper around `py.gaitkit`.

## Usage

```matlab
addpath('recode/matlab')
methods = gaitkit.listMethods();
res = gaitkit.detect('bayesian_bis', frames, 100, struct('position','mm','angles','deg'));
```

`frames` must be a struct array compatible with
`gaitkit.detect_events_structured` fields.

## Install backend from MATLAB

```matlab
gaitkit.installPythonBackend();
```
