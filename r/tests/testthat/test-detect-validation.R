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
  expect_error(
    gk_detect("", method = "bike"),
    "'data' must be a list or a file path"
  )
  expect_error(
    gk_detect("   ", method = "bike"),
    "'data' must be a list or a file path"
  )
  expect_error(
    gk_detect(list(foo = 1), method = "bike"),
    "'data' list should contain at least 'angle_frames' or 'fps' fields"
  )
  expect_error(
    gk_detect(list(angle_frames = list(), fps = 100), angles = ""),
    "'angles' must be a non-empty file path when provided"
  )
  expect_error(
    gk_detect(list(angle_frames = list(), fps = 100), angles_align = "bad"),
    "'angles_align' must be one of"
  )
  expect_error(
    gk_detect(list(angle_frames = list(), fps = 100), require_core_markers = NA),
    "'require_core_markers' must be TRUE or FALSE"
  )
})

test_that("gk_detect forwards C3D angle options to backend", {
  env <- environment(gk_detect)
  old_mod <- get(".gk_module", envir = env)
  old_wrap <- get(".wrap_result", envir = env)
  on.exit({
    assign(".gk_module", old_mod, envir = env)
    assign(".wrap_result", old_wrap, envir = env)
  }, add = TRUE)

  assign(".gk_module", function() {
    list(
      detect = function(...) {
        list(...)
      }
    )
  }, envir = env)
  assign(".wrap_result", function(x) x, envir = env)

  out <- gk_detect(
    "trial.c3d",
    method = "bike",
    angles = "angles.mat",
    angles_align = "second_hs",
    require_core_markers = TRUE
  )

  expect_equal(out$data, "trial.c3d")
  expect_equal(out$method, "bike")
  expect_equal(out$angles, "angles.mat")
  expect_equal(out$angles_align, "second_hs")
  expect_true(isTRUE(out$require_core_markers))
})

test_that("gk_detect_ensemble validates numeric controls before Python bridge", {
  dummy <- list(angle_frames = list(), fps = 100)
  expect_error(
    gk_detect_ensemble(dummy, methods = "bike"),
    "'methods' must be a character vector with at least two methods"
  )
  expect_error(
    gk_detect_ensemble(dummy, methods = c("bike", "   ")),
    "'methods' cannot contain empty entries"
  )
  expect_error(
    gk_detect_ensemble(dummy, methods = c("bike", "bike")),
    "'methods' cannot contain duplicates"
  )
  expect_error(
    gk_detect_ensemble(dummy, methods = c("bike", "zeni"), min_votes = 0),
    "'min_votes' must be an integer >= 1"
  )
  expect_error(
    gk_detect_ensemble(dummy, methods = c("bike", "zeni"), min_votes = 1.5),
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
  expect_error(
    gk_detect_ensemble("", methods = c("bike", "zeni")),
    "'data' must be a list or a file path"
  )
  expect_error(
    gk_detect_ensemble("   ", methods = c("bike", "zeni")),
    "'data' must be a list or a file path"
  )
  expect_error(
    gk_detect_ensemble(list(foo = 1), methods = c("bike", "zeni")),
    "'data' list should contain at least 'angle_frames' or 'fps' fields"
  )
})

test_that("gk_load_example validates name before Python bridge", {
  expect_error(
    gk_load_example(""),
    "'name' must be a non-empty character scalar"
  )
  expect_error(
    gk_load_example("   "),
    "'name' must be a non-empty character scalar"
  )
  expect_error(
    gk_load_example(1),
    "'name' must be a non-empty character scalar"
  )
})
