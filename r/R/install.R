#' Install Python Backend (gaitkit)
#'
#' Convenience helper to install the Python package `gaitkit` in the
#' active or selected reticulate environment.
#'
#' @param envname Optional environment name passed to \code{reticulate::py_install}.
#' @param pip Logical, whether to use pip. Defaults to TRUE.
#' @param package Python package specifier passed to \code{reticulate::py_install}.
#'   Defaults to \code{"gaitkit"}. Example: \code{"gaitkit[all]"}.
#' @return Invisible TRUE on success.
#' @export
gk_install_python <- function(envname = NULL, pip = TRUE, package = "gaitkit") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required", call. = FALSE)
  }
  if (!is.null(envname) && (!is.character(envname) || length(envname) != 1L || !nzchar(envname))) {
    stop("'envname' must be NULL or a non-empty character scalar", call. = FALSE)
  }
  if (!is.logical(pip) || length(pip) != 1L || is.na(pip)) {
    stop("'pip' must be TRUE or FALSE", call. = FALSE)
  }
  if (!is.character(package) || length(package) != 1L || !nzchar(trimws(package))) {
    stop("'package' must be a non-empty character scalar", call. = FALSE)
  }
  package <- trimws(package)

  tryCatch(
    reticulate::py_install(package, envname = envname, pip = pip),
    error = function(e) {
      stop(
        paste0(
          "Failed to install Python package '", package, "' via reticulate::py_install(). ",
          "Check network access and your Python environment configuration. ",
          "Original error: ", conditionMessage(e)
        ),
        call. = FALSE
      )
    }
  )
  invisible(TRUE)
}
