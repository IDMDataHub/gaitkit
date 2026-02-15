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
