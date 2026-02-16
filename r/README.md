# gaitkit (R interface)

This package exposes the Python `gaitkit` backend in R via `reticulate`.
Current status: CRAN submission in progress.

## Install (development)

```r
install.packages(c("reticulate", "jsonlite", "devtools"))

# From repository root:
devtools::install_local("r")

# Install Python backend in the active reticulate environment:
library(gaitkit)
gk_install_python()
gk_install_python(package = "gaitkit[all]") # optional extras
```

For reproducible setups, pin the Python interpreter before calling gaitkit:

```r
reticulate::use_python("/path/to/python", required = TRUE)
```

In non-interactive sessions (CI, Rscript), exporting `RETICULATE_PYTHON`
before starting R is the most reliable option.

Reproducibility checklist: see `REPRODUCIBILITY.md` at repository root.

## Run tests

```bash
R_LIBS_USER=/tmp/gaitkit-r-lib \
RETICULATE_PYTHON=/tmp/gaitkit-r-venv/bin/python \
Rscript -e "testthat::test_dir('r/tests/testthat', reporter='summary')"
```

## Build and check package

```bash
cd r
R_LIBS_USER=/tmp/gaitkit-r-lib RETICULATE_PYTHON=/tmp/gaitkit-r-venv/bin/python R CMD build .
R_LIBS_USER=/tmp/gaitkit-r-lib RETICULATE_PYTHON=/tmp/gaitkit-r-venv/bin/python R CMD check gaitkit_0.1.0.tar.gz --no-manual
```

## Usage

```r
library(gaitkit)

methods <- gk_methods()
trial <- gk_load_example("healthy")
res <- gk_detect(trial, method = "bike")
print(res)
```

If R cannot find the Python module, set the interpreter first:

```r
reticulate::use_python("/path/to/python", required = TRUE)
```
