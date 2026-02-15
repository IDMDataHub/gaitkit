# Reproducibility Checklist

This checklist documents the minimum steps to reproduce core `gaitkit` outputs.

## 1. Environment

- Python: 3.9+
- R: 4.1+ (optional wrapper)
- MATLAB: R2021b+ (optional wrapper)

## 2. Python install

```bash
python -m pip install -e ./python
```

Optional extras:

```bash
python -m pip install -e "./python[onnx,c3d,viz,deep]"
```

## 3. Deterministic smoke test on bundled examples

```bash
PYTHONPATH=python/src python -m unittest -v tests.test_api tests.test_bikegait
```

`tests/test_bikegait.py` validates that BIKE outputs on bundled examples match
`tests/expected_example_bike_regression.json`.

Detector-registry robustness smoke test:

```bash
PYTHONPATH=python/src python -m unittest -v tests.test_detectors_registry
```

Evaluation-module guardrail smoke tests:

```bash
PYTHONPATH=python/src python -m unittest -v \
  tests.test_evaluation_matching \
  tests.test_evaluation_metrics \
  tests.test_evaluation_statistics
```

## 4. R wrapper smoke test

```bash
Rscript -e "install.packages(c('testthat','reticulate','jsonlite'), repos='https://cloud.r-project.org')"
R_LIBS_USER=/tmp/gaitkit-r-lib \
RETICULATE_PYTHON=/tmp/gaitkit-r-venv/bin/python \
Rscript -e "testthat::test_dir('r/tests/testthat', reporter='summary')"
```

## 5. C3D external validation (recommended before release)

- Run `load_c3d()` on representative real-world C3D files.
- Validate event timing against known annotations/force-plate references.
- Document marker convention used (`pig` vs `isb`) and sampling rate.

## 6. Release gate

- All unit tests pass.
- Bundled regression file unchanged unless intentional algorithm updates.
- Package metadata points to active repository and issue tracker.
- Citation metadata (`CITATION.cff`) is present and up to date.

## 7. Packaging checks (pre-release)

Python sdist + wheel:

```bash
cd python
python -m pip install build
python -m build
```

R source package + check:

```bash
cd r
R_LIBS_USER=/tmp/gaitkit-r-lib RETICULATE_PYTHON=/tmp/gaitkit-r-venv/bin/python R CMD build .
R_LIBS_USER=/tmp/gaitkit-r-lib RETICULATE_PYTHON=/tmp/gaitkit-r-venv/bin/python R CMD check gaitkit_0.1.0.tar.gz --no-manual
```
