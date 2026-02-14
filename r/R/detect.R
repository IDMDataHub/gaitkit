#' List Available Detection Methods
#'
#' @return Character vector of method names accepted by \code{gk_detect}.
#'
#' @examples
#' gk_methods()
#' # [1] "bike" "zeni" "oconnor" ...
#'
#' @export
gk_methods <- function() {
  gk <- .gk_module()
  reticulate::py_to_r(gk$list_methods())
}

#' Detect Gait Events
#'
#' Run a gait event detector on motion capture data.
#'
#' @param data Input data. A named list with \code{angle_frames} and \code{fps}
#'   (as returned by \code{gk_load_example}), or a path to a C3D file.
#' @param method Character, detection method (default \code{"bike"}).
#'   See \code{gk_methods()} for all options.
#' @param fps Numeric, sampling frequency in Hz. Required unless \code{data}
#'   already contains fps information.
#'
#' @return An S3 object of class \code{gaitkit_result} with elements:
#'   \describe{
#'     \item{left_hs}{Left heel-strikes (data.frame: frame, time)}
#'     \item{right_hs}{Right heel-strikes}
#'     \item{left_to}{Left toe-offs}
#'     \item{right_to}{Right toe-offs}
#'     \item{events}{All events as a data.frame}
#'     \item{cycles}{Gait cycles data.frame}
#'     \item{method}{Method name}
#'     \item{fps}{Sampling frequency}
#'     \item{n_frames}{Number of input frames}
#'   }
#'
#' @examples
#' trial <- gk_load_example("healthy")
#' result <- gk_detect(trial)
#' print(result)
#' summary(result)
#'
#' @export
gk_detect <- function(data, method = "bike", fps = NULL) {
  gk <- .gk_module()

  if (is.character(data) && length(data) == 1L) {
    py_data <- data
  } else if (is.list(data)) {
    py_json <- reticulate::import("json", convert = FALSE)
    data_json <- jsonlite::toJSON(data, auto_unbox = TRUE, null = "null",
                                  digits = 6)
    py_data <- py_json$loads(data_json)
  } else {
    stop("'data' must be a list or a file path", call. = FALSE)
  }

  kwargs <- list(data = py_data, method = method)
  if (!is.null(fps)) kwargs$fps <- as.numeric(fps)

  py_result <- do.call(gk$detect, kwargs)

  .wrap_result(py_result)
}

#' Ensemble Detection (Multi-Method Voting)
#'
#' Run multiple detectors and combine results via temporal clustering
#' and majority vote.
#'
#' @param data Input data (same as \code{gk_detect}).
#' @param methods Character vector of methods. Default: all training-free methods.
#' @param min_votes Integer, minimum number of agreeing methods (default 2).
#' @param tolerance_ms Numeric, temporal tolerance for matching in ms (default 50).
#' @param fps Numeric, sampling frequency.
#'
#' @return A \code{gaitkit_result} with confidence scores for each event.
#'
#' @examples
#' trial <- gk_load_example("healthy")
#' result <- gk_detect_ensemble(trial, methods = c("bike", "zeni", "oconnor"))
#'
#' @export
gk_detect_ensemble <- function(data, methods = NULL, min_votes = 2L,
                               tolerance_ms = 50, fps = NULL) {
  gk <- .gk_module()

  if (is.character(data) && length(data) == 1L) {
    py_data <- data
  } else if (is.list(data)) {
    py_json <- reticulate::import("json", convert = FALSE)
    data_json <- jsonlite::toJSON(data, auto_unbox = TRUE, null = "null",
                                  digits = 6)
    py_data <- py_json$loads(data_json)
  } else {
    stop("'data' must be a list or a file path", call. = FALSE)
  }

  kwargs <- list(data = py_data, min_votes = as.integer(min_votes),
                 tolerance_ms = as.numeric(tolerance_ms))
  if (!is.null(methods)) kwargs$methods <- as.list(methods)
  if (!is.null(fps)) kwargs$fps <- as.numeric(fps)

  py_result <- do.call(gk$detect_ensemble, kwargs)
  .wrap_result(py_result)
}

#' Load a Bundled Example Trial
#'
#' @param name Character. Available: "healthy", "parkinson", "kuopio", "stroke".
#'
#' @return A named list with \code{angle_frames}, \code{fps}, \code{n_frames},
#'   \code{description}, \code{source}, \code{doi}, \code{population}.
#'
#' @examples
#' trial <- gk_load_example("healthy")
#' cat(trial$description, "\n")
#' cat("Frames:", trial$n_frames, "at", trial$fps, "Hz\n")
#'
#' @export
gk_load_example <- function(name = "healthy") {
  gk <- .gk_module()
  py_trial <- gk$load_example(name)
  reticulate::py_to_r(py_trial)
}

#' List Available Example Datasets
#'
#' @return Character vector of example names.
#' @export
gk_list_examples <- function() {
  gk <- .gk_module()
  reticulate::py_to_r(gk$list_examples())
}

# ── Internal helpers ────────────────────────────────────────────────

.gk_module <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required", call. = FALSE)
  }
  if (!reticulate::py_module_available("gaitkit")) {
    stop(
      "Python module 'gaitkit' not found. Install with: pip install gaitkit",
      call. = FALSE
    )
  }
  reticulate::import("gaitkit", convert = FALSE)
}

.wrap_result <- function(py_result) {
  r <- reticulate::py_to_r(py_result)

  events_df <- tryCatch(
    reticulate::py_to_r(py_result$events),
    error = function(e) data.frame()
  )
  cycles_df <- tryCatch(
    reticulate::py_to_r(py_result$cycles),
    error = function(e) data.frame()
  )

  result <- list(
    left_hs  = r$left_hs,
    right_hs = r$right_hs,
    left_to  = r$left_to,
    right_to = r$right_to,
    events   = events_df,
    cycles   = cycles_df,
    method   = r$method,
    fps      = r$fps,
    n_frames = r$n_frames
  )
  class(result) <- "gaitkit_result"
  result
}
