# gaitkit (CRAN package skeleton)

This package exposes the `gaitkit` Python backend in R via `reticulate`.

## Install (local during development)

```r
# In R
install.packages(c("reticulate", "jsonlite"))
# install python package first (wheel or editable)
# then:
devtools::install_local("gaitkit/r")
```

## Usage

```r
library(gaitkit)
methods <- gk_methods()
trial <- gk_load_example("healthy")
res <- gk_detect(trial, method = "bike")
```

`frames` is a list of named lists compatible with
`gaitkit.detect`.
