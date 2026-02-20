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

test_that("proprietary payload is normalized to angle_frames trial", {
  payload <- list(
    meta = list(fps = 120),
    angles = list(
      frames = list(
        list(frame_idx = 10, hip_L = 1, knee_L = 2, ankle_L = 3),
        list(frame_idx = 11, hip_R = 4, knee_R = 5, ankle_R = 6)
      )
    )
  )
  out <- .normalize_input_data(payload, fps = NULL)
  expect_true(is.list(out))
  expect_true("angle_frames" %in% names(out))
  expect_true("fps" %in% names(out))
  expect_equal(out$fps, 120)
  expect_equal(out$angle_frames[[1]]$frame_index, 10)
  expect_equal(out$angle_frames[[1]]$left_hip_angle, 1)
  expect_equal(out$angle_frames[[2]]$right_knee_angle, 5)
})

test_that("proprietary JSON file path is supported and validated", {
  tf <- tempfile(fileext = ".json")
  payload <- list(
    meta = list(fps = 100),
    angles = list(
      frames = list(
        list(frame_idx = 0, hip_L = 1, knee_L = 2, ankle_L = 3)
      )
    )
  )
  jsonlite::write_json(payload, tf, auto_unbox = TRUE, pretty = TRUE)
  out <- .normalize_input_data(tf, fps = NULL)
  expect_true(is.list(out))
  expect_true("angle_frames" %in% names(out))
  expect_equal(out$fps, 100)

  bad <- tempfile(fileext = ".json")
  jsonlite::write_json(list(foo = 1), bad, auto_unbox = TRUE, pretty = TRUE)
  expect_error(
    .normalize_input_data(bad, fps = NULL),
    "Unsupported JSON input: expected a payload with angles.frames"
  )
})

test_that("gk_export_detection payload coercion accepts gaitkit_result", {
  x <- list(
    left_hs = list(list(frame = 10, time = 0.1, confidence = 0.9)),
    right_hs = list(),
    left_to = list(),
    right_to = list(list(frame = 20, time = 0.2, confidence = 0.8)),
    cycles = data.frame(
      cycle_id = 0L, side = "left", start_frame = 10L, toe_off_frame = 20L,
      end_frame = 30L, duration = 0.2, stance_percentage = 60
    ),
    method = "bike",
    fps = 100,
    n_frames = 1000L
  )
  class(x) <- "gaitkit_result"
  payload <- .coerce_export_payload(x)
  expect_true(all(c("meta", "heel_strikes", "toe_offs", "cycles") %in% names(payload)))
  expect_equal(payload$meta$detector, "bike")
  expect_equal(payload$heel_strikes[[1]]$frame_index, 10)
  expect_equal(payload$toe_offs[[1]]$frame_index, 20)
})
