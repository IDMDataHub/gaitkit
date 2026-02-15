if (!exists("print.gaitkit_result", mode = "function")) {
  source(file.path("..", "..", "R", "methods.R"))
}

test_that("print.gaitkit_result handles ensemble-style cycles", {
  x <- list(
    left_hs = list(list(frame = 1L, time = 0.01)),
    right_hs = list(list(frame = 2L, time = 0.02)),
    left_to = list(list(frame = 3L, time = 0.03)),
    right_to = list(list(frame = 4L, time = 0.04)),
    cycles = data.frame(duration = c(1.1, 1.0)),
    method = "ensemble",
    fps = 100,
    n_frames = 1000
  )
  class(x) <- "gaitkit_result"

  expect_invisible(print(x))
})

test_that("print.gaitkit_result validates input type", {
  expect_error(
    print.gaitkit_result("not_a_result"),
    "'x' must be a gaitkit_result list"
  )
})
