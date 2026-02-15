# Contributing to gaitkit

Thanks for contributing.

## Development workflow

1. Create a branch from `master`.
2. Make focused changes (one concern per commit when possible).
3. Run local tests before opening a PR.
4. Open a PR with a clear summary and validation notes.

## Local validation

### Python

```bash
python3 -m unittest -v
```

### R (reticulate integration)

```bash
R_LIBS_USER=/tmp/gaitkit-r-lib \
RETICULATE_PYTHON=/tmp/gaitkit-r-venv/bin/python \
Rscript -e "testthat::test_dir('r/tests/testthat', reporter='summary')"
```

## Code quality expectations

- Keep API behavior backward compatible unless a breaking change is explicit.
- Add or update tests for every behavior change.
- Validate error messages for invalid inputs (Python and R wrappers).
- Keep docs in sync with behavior and installation steps.

## Commit style

Preferred commit prefixes:

- `improve(...)`: behavior hardening/refactor without feature break.
- `test(...)`: tests added/updated.
- `docs(...)`: documentation updates.
- `fix(...)`: bug fix with user-visible behavior correction.
