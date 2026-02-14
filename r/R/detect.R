#' List Available Detection Methods
#'
#' @param module Python module name. Defaults to 'BIKEgait'.
#' @return Character vector of available methods.
#' @export
gait_methods <- function(module = "BIKEgait") {
  py_mod <- .gait_native_module(module)
  reticulate::py_to_r(py_mod$list_methods())
}

#' Detect Gait Events From Structured Frames
#'
#' @param method Detector name (e.g. 'bayesian_bis').
#' @param frames List of frame objects. Each frame is a named list with fields
#'   compatible with `BIKEgait.detect_events_structured`.
#' @param fps Sampling frequency in Hz.
#' @param module Python module name. Defaults to 'BIKEgait'.
#' @param units Named list with `position` (`mm`/`m`) and `angles` (`deg`/`rad`).
#' @return A named list with fields `heel_strikes`, `toe_offs`, and `cycles`.
#' @export
gait_detect <- function(
  method,
  frames,
  fps = 100,
  module = "BIKEgait",
  units = list(position = "mm", angles = "deg")
) {
  if (!is.character(method) || length(method) != 1L) {
    stop("'method' must be a single character string", call. = FALSE)
  }
  if (!is.list(frames)) {
    stop("'frames' must be a list", call. = FALSE)
  }
  if (!is.numeric(fps) || length(fps) != 1L || !is.finite(fps) || fps <= 0) {
    stop("'fps' must be a positive scalar", call. = FALSE)
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required", call. = FALSE)
  }

  py_mod <- .gait_native_module(module)
  py_json <- reticulate::import("json", convert = FALSE)

  frames_json <- jsonlite::toJSON(frames, auto_unbox = TRUE, null = "null")
  units_json <- jsonlite::toJSON(units, auto_unbox = TRUE, null = "null")
  py_frames <- py_json$loads(frames_json)
  py_units <- py_json$loads(units_json)

  py_res <- py_mod$detect_events_structured(method, py_frames, as.numeric(fps), py_units)
  reticulate::py_to_r(py_res)
}

.gait_native_module <- function(module) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required", call. = FALSE)
  }
  if (!reticulate::py_module_available(module)) {
    stop(
      sprintf("Python module '%s' not found in current environment", module),
      call. = FALSE
    )
  }
  reticulate::import(module, convert = FALSE)
}
