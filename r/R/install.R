#' Install Python Backend (BIKEgait)
#'
#' Convenience helper to install the Python package `BIKEgait` in the
#' active or selected `reticulate` environment.
#'
#' @param package Python package name. Defaults to `"BIKEgait"`.
#' @param envname Optional environment name passed to `reticulate::py_install`.
#' @param pip Logical, whether to use pip. Defaults to `TRUE`.
#' @return Invisible `TRUE` on success.
#' @export
gait_install_python <- function(package = "BIKEgait", envname = NULL, pip = TRUE) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required", call. = FALSE)
  }
  reticulate::py_install(package, envname = envname, pip = pip)
  invisible(TRUE)
}
