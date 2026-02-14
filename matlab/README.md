# MATLAB wrapper (BIKEgait)

MATLAB wrapper around `py.BIKEgait`.

## Usage

```matlab
addpath('recode/matlab')
methods = BIKEgait.listMethods();
res = BIKEgait.detect('bayesian_bis', frames, 100, struct('position','mm','angles','deg'));
```

`frames` must be a struct array compatible with
`BIKEgait.detect_events_structured` fields.

## Install backend from MATLAB

```matlab
BIKEgait.installPythonBackend();
```
