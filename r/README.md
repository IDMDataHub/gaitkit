# gaitkit (CRAN package skeleton)

This package exposes the `gaitkit` Python backend in R via `reticulate`.

## Install (local during development)

```r
# In R
install.packages(c("reticulate", "jsonlite"))
# install python package first (wheel or editable)
# then:
devtools::install_local("recode/cran/gaitkit")
```

## Usage

```r
library(gaitkit)
methods <- gait_methods()
res <- gait_detect(
  "bayesian_bis",
  frames,
  fps = 100,
  units = list(position = "mm", angles = "deg")
)
```

`frames` is a list of named lists compatible with
`gaitkit.detect_events_structured`.
