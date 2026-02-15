test_that("core API works on bundled example", {
  skip_if_not_installed("reticulate")
  if (!reticulate::py_module_available("gaitkit")) {
    skip("Python gaitkit module is not available in this environment")
  }

  methods <- gk_methods()
  expect_true(is.character(methods))
  expect_true("bike" %in% methods)

  trial <- gk_load_example("healthy")
  expect_true(is.list(trial))
  expect_true("angle_frames" %in% names(trial))
  expect_true("fps" %in% names(trial))

  result <- gk_detect(trial, method = "bike")
  expect_true(is.list(result))
  expect_s3_class(result, "gaitkit_result")
  expect_true(nrow(result$events) > 0)
})
