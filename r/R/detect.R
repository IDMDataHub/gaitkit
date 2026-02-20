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
#' @param data Input data. One of:
#'   \itemize{
#'   \item a named list with \code{angle_frames} and optional \code{fps}
#'   (as returned by \code{gk_load_example}),
#'   \item a proprietary MyoGait-like payload list with \code{angles$frames},
#'   \item a file path to a C3D file,
#'   \item a file path to a proprietary JSON payload.
#'   }
#' @param method Character, detection method (default \code{"bike"}).
#'   See \code{gk_methods()} for all options.
#' @param fps Numeric, sampling frequency in Hz. Required unless \code{data}
#'   already contains fps information.
#' @param angles Optional path to external angles (MAT/CSV/JSON) when
#'   \code{data} is a C3D path.
#' @param angles_align Alignment mode for external angles on C3D timeline.
#'   One of \code{"auto"}, \code{"second_hs"}, \code{"first_hs"},
#'   \code{"none"}, \code{"resample"}.
#' @param require_core_markers Logical. If \code{TRUE}, enforce presence of
#'   core gait markers when computing proxy angles from marker-only C3D.
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
gk_detect <- function(
  data,
  method = "bike",
  fps = NULL,
  angles = NULL,
  angles_align = "auto",
  require_core_markers = FALSE
) {
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
  if (!is.null(angles)) {
    if (!is.character(angles) || length(angles) != 1L || !nzchar(trimws(angles))) {
      stop("'angles' must be a non-empty file path when provided", call. = FALSE)
    }
  }
  if (!is.character(angles_align) || length(angles_align) != 1L || !nzchar(trimws(angles_align))) {
    stop("'angles_align' must be a non-empty character scalar", call. = FALSE)
  }
  if (!angles_align %in% c("auto", "second_hs", "first_hs", "none", "resample")) {
    stop("'angles_align' must be one of: 'auto', 'second_hs', 'first_hs', 'none', 'resample'", call. = FALSE)
  }
  if (!is.logical(require_core_markers) || length(require_core_markers) != 1L || is.na(require_core_markers)) {
    stop("'require_core_markers' must be TRUE or FALSE", call. = FALSE)
  }

  norm_data <- .normalize_input_data(data, fps = fps)
  if (is.character(norm_data) && length(norm_data) == 1L && nzchar(trimws(norm_data))) {
    py_data <- norm_data
  } else if (is.list(norm_data)) {
    py_json <- reticulate::import("json", convert = FALSE)
    data_json <- jsonlite::toJSON(norm_data, auto_unbox = TRUE, null = "null",
                                  digits = 6)
    py_data <- py_json$loads(data_json)
  } else {
    stop("'data' must be a list or a file path", call. = FALSE)
  }

  gk <- .gk_module()

  kwargs <- list(data = py_data, method = method)
  if (!is.null(fps)) kwargs$fps <- as.numeric(fps)
  if (!is.null(angles)) kwargs$angles <- angles
  kwargs$angles_align <- angles_align
  kwargs$require_core_markers <- isTRUE(require_core_markers)

  py_result <- do.call(gk$detect, kwargs)

  .wrap_result(py_result)
}

#' Ensemble Detection (Multi-Method Voting)
#'
#' Run multiple detectors and combine results via temporal clustering
#' and majority vote.
#'
#' @param data Input data (same contract as \code{gk_detect}).
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
    if (anyDuplicated(methods)) {
      stop("'methods' cannot contain duplicates", call. = FALSE)
    }
  }
  if (!is.numeric(min_votes) || length(min_votes) != 1L || is.na(min_votes) ||
      min_votes < 1 || min_votes != floor(min_votes)) {
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

  norm_data <- .normalize_input_data(data, fps = fps)
  if (is.character(norm_data) && length(norm_data) == 1L && nzchar(trimws(norm_data))) {
    py_data <- norm_data
  } else if (is.list(norm_data)) {
    py_json <- reticulate::import("json", convert = FALSE)
    data_json <- jsonlite::toJSON(norm_data, auto_unbox = TRUE, null = "null",
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

#' Export Detection Results to Files
#'
#' Export a \code{gaitkit_result} (or a structured payload) to one or more
#' files using the Python compatibility exporter.
#'
#' @param result A \code{gaitkit_result} object returned by \code{gk_detect()}
#'   / \code{gk_detect_ensemble()}, or a structured payload list with
#'   \code{meta}, \code{heel_strikes}, \code{toe_offs}, and \code{cycles}.
#' @param output_prefix Character scalar path prefix for output files.
#' @param formats Character vector of formats. Supported:
#'   \code{"json"}, \code{"csv"}, \code{"xlsx"}, \code{"myogait"}.
#'
#' @return Named character vector with written file paths.
#'
#' @examples
#' \donttest{
#' trial <- gk_load_example("healthy")
#' res <- gk_detect(trial, method = "bike")
#' gk_export_detection(res, tempfile("gaitkit_out"), formats = c("json", "myogait"))
#' }
#'
#' @export
gk_export_detection <- function(result, output_prefix, formats = "json") {
  if (!is.character(output_prefix) || length(output_prefix) != 1L || !nzchar(trimws(output_prefix))) {
    stop("'output_prefix' must be a non-empty character scalar", call. = FALSE)
  }
  if (!is.character(formats) || length(formats) < 1L) {
    stop("'formats' must be a non-empty character vector", call. = FALSE)
  }
  formats <- unique(tolower(trimws(formats)))
  if (any(!nzchar(formats))) {
    stop("'formats' cannot contain empty entries", call. = FALSE)
  }

  payload <- .coerce_export_payload(result)
  py_json <- reticulate::import("json", convert = FALSE)
  payload_json <- jsonlite::toJSON(payload, auto_unbox = TRUE, null = "null", digits = 6)
  py_payload <- py_json$loads(payload_json)

  gk <- .gk_module()
  py_paths <- gk$export_detection(
    py_payload,
    output_prefix = output_prefix,
    formats = as.list(formats)
  )
  reticulate::py_to_r(py_paths)
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

`%||%` <- function(x, y) {
  if (is.null(x) || (length(x) == 1L && is.na(x))) y else x
}

.is_myogait_payload <- function(x) {
  is.list(x) &&
    ("angles" %in% names(x)) &&
    is.list(x$angles) &&
    ("frames" %in% names(x$angles)) &&
    is.list(x$angles$frames)
}

.normalize_myogait_frame <- function(frame, i) {
  if (!is.list(frame)) return(NULL)
  out <- list(
    frame_index = as.integer(frame$frame_idx %||% (i - 1L)),
    trunk_angle = frame$trunk_angle,
    pelvis_tilt = frame$pelvis_tilt,
    left_hip_angle = frame$hip_L,
    right_hip_angle = frame$hip_R,
    left_knee_angle = frame$knee_L,
    right_knee_angle = frame$knee_R,
    left_ankle_angle = frame$ankle_L,
    right_ankle_angle = frame$ankle_R
  )
  if (!is.null(frame$landmark_positions) && is.list(frame$landmark_positions)) {
    out$landmark_positions <- frame$landmark_positions
  }
  out
}

.myogait_to_trial <- function(payload, fps = NULL) {
  frames <- payload$angles$frames
  norm_frames <- Filter(
    Negate(is.null),
    lapply(seq_along(frames), function(i) .normalize_myogait_frame(frames[[i]], i))
  )
  resolved_fps <- NULL
  if (!is.null(payload$meta) && is.list(payload$meta) && !is.null(payload$meta$fps)) {
    resolved_fps <- suppressWarnings(as.numeric(payload$meta$fps))
  }
  if (is.null(resolved_fps) || is.na(resolved_fps) || resolved_fps <= 0) {
    if (!is.null(fps)) {
      resolved_fps <- as.numeric(fps)
    } else {
      resolved_fps <- NULL
    }
  }

  trial <- list(angle_frames = norm_frames)
  if (!is.null(resolved_fps) && !is.na(resolved_fps) && resolved_fps > 0) {
    trial$fps <- resolved_fps
  }
  trial
}

.normalize_input_data <- function(data, fps = NULL) {
  if (is.character(data) && length(data) == 1L) {
    data <- trimws(data)
    if (!nzchar(data)) {
      stop("'data' must be a list or a file path", call. = FALSE)
    }
    if (grepl("\\.json$", data, ignore.case = TRUE)) {
      if (!file.exists(data)) {
        stop("JSON input file does not exist: ", data, call. = FALSE)
      }
      payload <- jsonlite::fromJSON(data, simplifyVector = FALSE)
      if (!.is_myogait_payload(payload)) {
        stop("Unsupported JSON input: expected a payload with angles.frames", call. = FALSE)
      }
      return(.myogait_to_trial(payload, fps = fps))
    }
    return(data)
  }

  if (!is.list(data)) {
    stop("'data' must be a list or a file path", call. = FALSE)
  }
  if ("angle_frames" %in% names(data) || "fps" %in% names(data)) {
    return(data)
  }
  if (.is_myogait_payload(data)) {
    return(.myogait_to_trial(data, fps = fps))
  }
  stop(
    "'data' list should contain at least 'angle_frames' or 'fps' fields",
    call. = FALSE
  )
}

.coerce_export_payload <- function(x) {
  if (!is.list(x)) {
    stop("'result' must be a gaitkit_result or a structured payload list", call. = FALSE)
  }
  if (all(c("meta", "heel_strikes", "toe_offs", "cycles") %in% names(x))) {
    return(x)
  }
  if (!inherits(x, "gaitkit_result")) {
    stop("'result' must be a gaitkit_result or a structured payload list", call. = FALSE)
  }

  hs_rows <- data.frame()
  if (is.list(x$left_hs) && length(x$left_hs) > 0) {
    hs_rows <- rbind(hs_rows, .events_list_to_rows(x$left_hs, "left"))
  }
  if (is.list(x$right_hs) && length(x$right_hs) > 0) {
    hs_rows <- rbind(hs_rows, .events_list_to_rows(x$right_hs, "right"))
  }

  to_rows <- data.frame()
  if (is.list(x$left_to) && length(x$left_to) > 0) {
    to_rows <- rbind(to_rows, .events_list_to_rows(x$left_to, "left"))
  }
  if (is.list(x$right_to) && length(x$right_to) > 0) {
    to_rows <- rbind(to_rows, .events_list_to_rows(x$right_to, "right"))
  }

  list(
    meta = list(
      detector = x$method %||% NA_character_,
      fps_hz = x$fps %||% NA_real_,
      n_frames = x$n_frames %||% NA_integer_
    ),
    heel_strikes = .rows_to_payload_list(hs_rows),
    toe_offs = .rows_to_payload_list(to_rows),
    cycles = if (is.data.frame(x$cycles)) .rows_to_payload_list(x$cycles) else list()
  )
}

.events_list_to_rows <- function(events, side) {
  do.call(rbind, lapply(events, function(ev) {
    data.frame(
      frame_index = as.integer(ev$frame %||% ev$frame_index %||% NA_integer_),
      time_s = as.numeric(ev$time %||% ev$time_s %||% NA_real_),
      side = as.character(side),
      confidence = as.numeric(ev$confidence %||% 1.0)
    )
  }))
}

.rows_to_payload_list <- function(df) {
  if (!is.data.frame(df) || nrow(df) == 0) return(list())
  rows <- vector("list", nrow(df))
  for (i in seq_len(nrow(df))) {
    row <- as.list(df[i, , drop = FALSE])
    rows[[i]] <- lapply(row, function(v) if (length(v) == 1L) v[[1]] else v)
  }
  rows
}

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
