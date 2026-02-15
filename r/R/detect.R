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
  if (is.character(method) && length(method) == 1L) {
    method <- trimws(method)
  }
  if (!is.character(method) || length(method) != 1L || !nzchar(method)) {
    stop("'method' must be a non-empty character scalar", call. = FALSE)
  }
  if (!is.null(fps)) {
    if (!is.numeric(fps) || length(fps) != 1L || is.na(fps) || fps <= 0) {
      stop("'fps' must be a positive numeric scalar", call. = FALSE)
    }
  }

  if (is.character(data) && length(data) == 1L && nzchar(trimws(data))) {
    py_data <- data
  } else if (is.list(data)) {
    if (!("angle_frames" %in% names(data)) && !("fps" %in% names(data))) {
      stop(
        "'data' list should contain at least 'angle_frames' or 'fps' fields",
        call. = FALSE
      )
    }
    py_json <- reticulate::import("json", convert = FALSE)
    data_json <- jsonlite::toJSON(data, auto_unbox = TRUE, null = "null",
                                  digits = 6)
    py_data <- py_json$loads(data_json)
  } else {
    stop("'data' must be a list or a file path", call. = FALSE)
  }

  gk <- .gk_module()

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
  if (!is.null(methods)) {
    if (!is.character(methods) || length(methods) < 2L) {
      stop("'methods' must be a character vector with at least two methods", call. = FALSE)
    }
    methods <- trimws(methods)
    if (any(!nzchar(methods))) {
      stop("'methods' cannot contain empty entries", call. = FALSE)
    }
  }
  if (!is.numeric(min_votes) || length(min_votes) != 1L || is.na(min_votes) || min_votes < 1) {
    stop("'min_votes' must be an integer >= 1", call. = FALSE)
  }
  if (!is.numeric(tolerance_ms) || length(tolerance_ms) != 1L || is.na(tolerance_ms) || tolerance_ms < 0) {
    stop("'tolerance_ms' must be a numeric value >= 0", call. = FALSE)
  }
  if (!is.null(fps)) {
    if (!is.numeric(fps) || length(fps) != 1L || is.na(fps) || fps <= 0) {
      stop("'fps' must be a positive numeric scalar", call. = FALSE)
    }
  }

  if (is.character(data) && length(data) == 1L && nzchar(trimws(data))) {
    py_data <- data
  } else if (is.list(data)) {
    if (!("angle_frames" %in% names(data)) && !("fps" %in% names(data))) {
      stop(
        "'data' list should contain at least 'angle_frames' or 'fps' fields",
        call. = FALSE
      )
    }
    py_json <- reticulate::import("json", convert = FALSE)
    data_json <- jsonlite::toJSON(data, auto_unbox = TRUE, null = "null",
                                  digits = 6)
    py_data <- py_json$loads(data_json)
  } else {
    stop("'data' must be a list or a file path", call. = FALSE)
  }

  gk <- .gk_module()

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
  if (is.character(name) && length(name) == 1L) {
    name <- trimws(name)
  }
  if (!is.character(name) || length(name) != 1L || !nzchar(name)) {
    stop("'name' must be a non-empty character scalar", call. = FALSE)
  }
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
  has_attr <- function(name) {
    isTRUE(reticulate::py_has_attr(py_result, name))
  }
  get_attr <- function(name, default = NULL) {
    if (!has_attr(name)) return(default)
    reticulate::py_to_r(py_result[[name]])
  }

  # Standard single-detector result shape
  if (has_attr("left_hs") && has_attr("right_hs") &&
      has_attr("left_to") && has_attr("right_to")) {
    events_df <- tryCatch(
      reticulate::py_to_r(py_result$events),
      error = function(e) data.frame()
    )
    cycles_df <- tryCatch(
      reticulate::py_to_r(py_result$cycles),
      error = function(e) data.frame()
    )

    result <- list(
      left_hs  = get_attr("left_hs", list()),
      right_hs = get_attr("right_hs", list()),
      left_to  = get_attr("left_to", list()),
      right_to = get_attr("right_to", list()),
      events   = events_df,
      cycles   = cycles_df,
      method   = get_attr("method", NA_character_),
      fps      = get_attr("fps", NA_real_),
      n_frames = get_attr("n_frames", NA_integer_)
    )
    class(result) <- "gaitkit_result"
    return(result)
  }

  # Ensemble result shape (heel_strikes / toe_offs)
  hs_raw <- tryCatch(reticulate::py_to_r(py_result$heel_strikes), error = function(e) list())
  to_raw <- tryCatch(reticulate::py_to_r(py_result$toe_offs), error = function(e) list())
  fps <- as.numeric(get_attr("fps", NA_real_))

  .norm_event <- function(ev, event_type) {
    frame <- if (!is.null(ev$frame)) ev$frame else ev$frame_index
    time <- if (!is.null(ev$time)) ev$time else if (!is.na(fps)) frame / fps else NA_real_
    conf <- if (!is.null(ev$confidence)) ev$confidence else 1.0
    list(
      frame = as.integer(frame),
      time = as.numeric(time),
      side = as.character(ev$side),
      confidence = as.numeric(conf),
      event_type = event_type
    )
  }

  hs_norm <- lapply(hs_raw, .norm_event, event_type = "HS")
  to_norm <- lapply(to_raw, .norm_event, event_type = "TO")
  all_events <- c(hs_norm, to_norm)
  if (length(all_events) > 0) {
    events_df <- do.call(rbind, lapply(all_events, as.data.frame))
    events_df <- events_df[order(events_df$time), , drop = FALSE]
    rownames(events_df) <- NULL
  } else {
    events_df <- data.frame()
  }

  left_hs <- lapply(Filter(function(e) identical(e$side, "left"), hs_norm), function(e) {
    list(frame = e$frame, time = e$time, confidence = e$confidence)
  })
  right_hs <- lapply(Filter(function(e) identical(e$side, "right"), hs_norm), function(e) {
    list(frame = e$frame, time = e$time, confidence = e$confidence)
  })
  left_to <- lapply(Filter(function(e) identical(e$side, "left"), to_norm), function(e) {
    list(frame = e$frame, time = e$time, confidence = e$confidence)
  })
  right_to <- lapply(Filter(function(e) identical(e$side, "right"), to_norm), function(e) {
    list(frame = e$frame, time = e$time, confidence = e$confidence)
  })

  cycles_raw <- tryCatch(reticulate::py_to_r(py_result$cycles), error = function(e) list())
  cycles_df <- if (length(cycles_raw) > 0) as.data.frame(cycles_raw) else data.frame()

  result <- list(
    left_hs = left_hs,
    right_hs = right_hs,
    left_to = left_to,
    right_to = right_to,
    events = events_df,
    cycles = cycles_df,
    method = get_attr("method", "ensemble"),
    fps = fps,
    n_frames = get_attr("n_frames", NA_integer_)
  )
  class(result) <- "gaitkit_result"
  result
}
