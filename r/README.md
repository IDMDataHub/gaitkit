# gaitkit (R interface)

This package exposes the Python `gaitkit` backend in R via `reticulate`.

## Install (development)

```r
install.packages(c("reticulate", "jsonlite"))

# Install Python backend first (wheel or editable), then:
devtools::install_local("gaitkit/r")
```

## Usage

```r
library(gaitkit)

methods <- gk_methods()
trial <- gk_load_example("healthy")
res <- gk_detect(trial, method = "bike")
print(res)
```
