if (!exists("gk_install_python", mode = "function")) {
  source(file.path("..", "..", "R", "install.R"))
}

test_that("gk_install_python validates arguments before reticulate call", {
  skip_if_not_installed("reticulate")

  expect_error(
    gk_install_python(envname = ""),
    "'envname' must be NULL or a non-empty character scalar"
  )
  expect_error(
    gk_install_python(envname = 1),
    "'envname' must be NULL or a non-empty character scalar"
  )
  expect_error(
    gk_install_python(pip = NA),
    "'pip' must be TRUE or FALSE"
  )
  expect_error(
    gk_install_python(package = ""),
    "'package' must be a non-empty character scalar"
  )
  expect_error(
    gk_install_python(package = 1),
    "'package' must be a non-empty character scalar"
  )
})
