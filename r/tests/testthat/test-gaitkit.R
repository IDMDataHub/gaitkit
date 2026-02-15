test_that("core API works on bundled example", {
  skip_if_not_installed("reticulate")
  if (!reticulate::py_module_available("gaitkit")) {
    skip("Python gaitkit module is not available in this environment")
  }

  methods <- gaitkit::gk_methods()
  expect_true(is.character(methods))
  expect_true("bike" %in% methods)

  trial <- gaitkit::gk_load_example("healthy")
  expect_true(is.list(trial))
  expect_true("angle_frames" %in% names(trial))
  expect_true("fps" %in% names(trial))

  result <- gaitkit::gk_detect(trial, method = "bike")
  expect_true(is.list(result))
  expect_s3_class(result, "gaitkit_result")
  expect_true(nrow(result$events) > 0)
})

test_that("example listing and ensemble API are reachable", {
  skip_if_not_installed("reticulate")
  if (!reticulate::py_module_available("gaitkit")) {
    skip("Python gaitkit module is not available in this environment")
  }

  ex <- gaitkit::gk_list_examples()
  expect_true(is.character(ex))
  expect_true("healthy" %in% ex)

  trial <- gaitkit::gk_load_example("healthy")
  ens <- gaitkit::gk_detect_ensemble(
    trial,
    methods = c("bayesian_bis", "zeni", "oconnor"),
    min_votes = 2L
  )
  expect_s3_class(ens, "gaitkit_result")
  expect_true(is.data.frame(ens$events))
  if (nrow(ens$events) > 0) {
    expect_true(all(c("frame", "time", "side", "event_type") %in% names(ens$events)))
  }
})
