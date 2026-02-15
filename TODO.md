# TODO - Publication & Community Readiness

This list tracks remaining work to bring `gaitkit` to a fully production-grade
state for publication and broad community usage.

## 1) CI and Quality Gates

- [ ] Add GitHub Actions CI for Python (`3.9` to `3.12`) on Linux/macOS/Windows.
- [ ] Add GitHub Actions CI for R (`release` + `devel`) with `reticulate` checks.
- [ ] Add a matrix job that validates Python-R bridge end-to-end.
- [ ] Add required status checks on `master` before merge.
- [ ] Add coverage upload and minimum coverage threshold policy.

## 2) Static Analysis and Style

- [ ] Add Python linting (`ruff` or `flake8`) and formatting policy.
- [ ] Add Python type-checking (`mypy`/`pyright`) for public API modules.
- [ ] Add R quality checks (`R CMD check`, `lintr`) in CI.
- [ ] Add MATLAB lint/check script for wrapper files.

## 3) Packaging and Release Process

- [x] Add release checklist (version bump, changelog, tag, artifacts, notes).
- [ ] Publish and test source/wheel builds on a clean environment.
- [ ] Add CRAN submission checklist and reverse-dependency smoke checks.
- [ ] Ensure package metadata URLs are publicly reachable before CRAN/PyPI submission.
- [ ] Add compatibility table (Python/R/MATLAB versions + tested OS matrix).
- [ ] Add signed release tags and reproducible release notes template.

## 4) Documentation for External Users

- [x] Add `CONTRIBUTING.md` (bug reports, coding standards, test policy).
- [x] Add `SECURITY.md` (reporting channel and disclosure policy).
- [x] Add explicit troubleshooting section for reticulate interpreter selection.
- [ ] Add "minimal reproducible example" templates for issue reports.
- [ ] Add benchmark page (speed + accuracy) with dataset and hardware details.

## 5) Validation on Real C3D Data

- [ ] Build a cross-dataset C3D regression suite (healthy + pathological gait).
- [ ] Add a reference-results snapshot test set for non-regression tracking.
- [ ] Add force-plate-zone-aware scoring examples in user docs.
- [ ] Validate behavior on missing markers/noisy trajectories per method.
- [ ] Publish expected uncertainty bounds per method and condition.

## 6) API Stability and Deprecation

- [ ] Define API stability policy (`0.x` -> `1.0` transition criteria).
- [ ] Add deprecation warnings and migration notes for renamed APIs.
- [ ] Lock JSON schema contracts for CLI and MATLAB structured inputs.
- [ ] Add changelog sections for breaking vs non-breaking changes.

## 7) Performance and Scalability

- [ ] Add performance regression tests for C-native and Python fallback paths.
- [ ] Add large-trial stress tests (long sequences, batch usage).
- [ ] Profile hot paths and document optimization decisions.
- [ ] Add memory-usage checks for long recordings.
