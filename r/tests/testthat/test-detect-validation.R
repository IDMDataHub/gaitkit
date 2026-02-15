if (!exists("gk_detect", mode = "function")) {
  source(file.path("..", "..", "R", "detect.R"))
}

test_that("gk_detect validates method and fps before Python bridge", {
  expect_error(
    gk_detect(list(angle_frames = list(), fps = 100), method = ""),
    "'method' must be a non-empty character scalar"
  )
  expect_error(
    gk_detect(list(angle_frames = list(), fps = 100), method = "bike", fps = 0),
    "'fps' must be a positive numeric scalar"
  )
})

test_that("gk_detect_ensemble validates numeric controls before Python bridge", {
  dummy <- list(angle_frames = list(), fps = 100)
  expect_error(
    gk_detect_ensemble(dummy, methods = "bike"),
    "'methods' must be a character vector with at least two methods"
  )
  expect_error(
    gk_detect_ensemble(dummy, methods = c("bike", "zeni"), min_votes = 0),
    "'min_votes' must be an integer >= 1"
  )
  expect_error(
    gk_detect_ensemble(dummy, methods = c("bike", "zeni"), tolerance_ms = -1),
    "'tolerance_ms' must be a numeric value >= 0"
  )
  expect_error(
    gk_detect_ensemble(dummy, methods = c("bike", "zeni"), fps = 0),
    "'fps' must be a positive numeric scalar"
  )
})
