# gaitkit (R interface)

This package exposes the Python `gaitkit` backend in R via `reticulate`.

## Install (development)

```r
install.packages(c("reticulate", "jsonlite", "devtools"))

# From repository root:
devtools::install_local("r")

# Install Python backend in the active reticulate environment:
library(gaitkit)
gk_install_python()
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
