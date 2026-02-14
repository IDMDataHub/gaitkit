#' Install Python Backend (gaitkit)
#'
#' Convenience helper to install the Python package `gaitkit` in the
#' active or selected reticulate environment.
#'
#' @param envname Optional environment name passed to \code{reticulate::py_install}.
#' @param pip Logical, whether to use pip. Defaults to TRUE.
#' @return Invisible TRUE on success.
#' @export
gk_install_python <- function(envname = NULL, pip = TRUE) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required", call. = FALSE)
  }
  reticulate::py_install("gaitkit", envname = envname, pip = pip)
  invisible(TRUE)
}
