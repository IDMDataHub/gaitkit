# Release and Citation Plan

This document tracks the pre-publication release flow for `gaitkit`.

## Current policy

- `gaitkit` is cited as **software** (not as a journal paper).
- BIKE manuscript references are not listed as published journal references
  until submission/acceptance.
- `CITATION.cff` is the source of truth for citation metadata.

## Pre-release checklist

1. Run full local non-regression:
   - Python: `python3 -m unittest discover -s tests -q`
   - R: `R_LIBS_USER=/tmp/gaitkit-r-lib RETICULATE_PYTHON=/tmp/gaitkit-r-venv/bin/python Rscript -e "testthat::test_dir('r/tests/testthat', reporter='summary')"`
2. Build artifacts:
   - Python: `cd python && python -m pip install build twine && python -m build && twine check dist/*`
   - R: `cd r && R CMD build . && R CMD check gaitkit_0.1.0.tar.gz --as-cran`
3. Tag pre-release (`vX.Y.Z-rcN`) and publish release notes.

## Before public PyPI/CRAN submission

1. Ensure the project URL used in package metadata is publicly reachable.
   - If the repository is private, publish a public mirror or make it public
     before CRAN/PyPI submission (URL checks will fail otherwise).
2. Mint a software DOI (recommended: Zenodo + GitHub release integration).
3. Add DOI in:
   - `CITATION.cff`
   - `README.md` citation section
4. Keep BIKE as software citation unless/until journal publication is public.

## Trusted Publishing setup (PyPI + TestPyPI)

`release.yml` supports:
- tag `v*` -> publish to PyPI,
- manual run (`workflow_dispatch`) -> publish to TestPyPI.

Configure trusted publishers on both services:

1. PyPI (`https://pypi.org/manage/project/gaitkit/publishing/`)
   - Owner: `IDMDataHub`
   - Repository: `gaitkit`
   - Workflow name: `release.yml`
2. TestPyPI (`https://test.pypi.org/manage/project/gaitkit/publishing/`)
   - Owner: `IDMDataHub`
   - Repository: `gaitkit`
   - Workflow name: `release.yml`

Publish flow:

1. Manual dry run to TestPyPI from GitHub Actions (`Release` workflow).
2. Validate installation from TestPyPI.
3. Create and push release tag (`vX.Y.Z`) to publish on PyPI.
