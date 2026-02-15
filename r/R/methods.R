#' Print a gaitkit_result
#'
#' @param x A \code{gaitkit_result} object.
#' @param ... Ignored.
#' @export
print.gaitkit_result <- function(x, ...) {
  if (!is.list(x)) {
    stop("'x' must be a gaitkit_result list", call. = FALSE)
  }
  left_hs <- if (is.null(x$left_hs)) list() else x$left_hs
  right_hs <- if (is.null(x$right_hs)) list() else x$right_hs
  left_to <- if (is.null(x$left_to)) list() else x$left_to
  right_to <- if (is.null(x$right_to)) list() else x$right_to

  method_raw <- x$method
  if (is.null(method_raw) || length(method_raw) != 1L) {
    method <- "unknown"
  } else {
    method <- as.character(method_raw)
    if (!nzchar(method)) method <- "unknown"
  }
  fps <- suppressWarnings(as.numeric(x$fps))
  n_frames <- suppressWarnings(as.integer(x$n_frames))
  if (length(fps) != 1L) fps <- NA_real_
  if (length(n_frames) != 1L) n_frames <- NA_integer_
  if (!is.finite(fps)) fps <- NA_real_
  if (is.na(n_frames)) n_frames <- NA_integer_

  n_hs <- length(left_hs) + length(right_hs)
  n_to <- length(left_to) + length(right_to)
  cat(sprintf("gaitkit_result  method=%s  fps=%s  frames=%s\n",
              method,
              if (is.finite(fps)) sprintf("%.0f", fps) else "NA",
              if (!is.na(n_frames)) as.character(n_frames) else "NA"))
  cat(sprintf("  Heel-strikes: %d (L=%d, R=%d)\n",
              n_hs, length(left_hs), length(right_hs)))
  cat(sprintf("  Toe-offs:     %d (L=%d, R=%d)\n",
              n_to, length(left_to), length(right_to)))
  if (is.data.frame(x$cycles) && nrow(x$cycles) > 0) {
    stride_col <- if ("stride_time" %in% names(x$cycles)) "stride_time" else
      if ("duration" %in% names(x$cycles)) "duration" else NULL
    if (!is.null(stride_col)) {
      st <- mean(x$cycles[[stride_col]], na.rm = TRUE)
      cad <- if ("cadence" %in% names(x$cycles)) {
        mean(x$cycles$cadence, na.rm = TRUE)
      } else {
        if (is.finite(st) && st > 0) 60 / st else NA_real_
      }
      if (is.finite(st) && is.finite(cad)) {
        cat(sprintf("  Stride time:  %.3f s (cadence %.0f steps/min)\n", st, cad))
      }
    }
  }
  invisible(x)
}

#' Summary of a gaitkit_result
#'
#' @param object A \code{gaitkit_result} object.
#' @param ... Ignored.
#' @export
summary.gaitkit_result <- function(object, ...) {
  if (!is.list(object)) {
    stop("'object' must be a gaitkit_result list", call. = FALSE)
  }
  print.gaitkit_result(object)
  if (is.data.frame(object$cycles) && nrow(object$cycles) > 0) {
    cat("\nCycles:\n")
    print(object$cycles)
  }
  invisible(object)
}

#' Plot Gait Events
#'
#' @param x A \code{gaitkit_result} object.
#' @param type Character: "events" (default) for timeline, "cycles" for butterfly.
#' @param ... Additional arguments passed to \code{plot}.
#'
#' @export
plot.gaitkit_result <- function(x, type = "events", ...) {
  if (!is.list(x)) {
    stop("'x' must be a gaitkit_result list", call. = FALSE)
  }
  if (!is.character(type) || length(type) != 1L || !(type %in% c("events", "cycles"))) {
    stop("'type' must be either 'events' or 'cycles'", call. = FALSE)
  }
  events <- x$events
  if (!is.data.frame(events) || nrow(events) == 0) {
    message("No events to plot.")
    return(invisible(NULL))
  }

  if (type == "events") {
    et <- tolower(as.character(events$event_type))
    hs <- events[et %in% c("hs", "heel_strike"), , drop = FALSE]
    to <- events[et %in% c("to", "toe_off"), , drop = FALSE]

    xlim <- range(events$time)
    ylim <- c(-0.5, 1.5)

    method_label <- x$method
    if (is.null(method_label) || length(method_label) != 1L) method_label <- "unknown"
    plot(xlim, ylim, type = "n", xlab = "Time (s)", ylab = "",
         main = paste("Gait Events -", as.character(method_label)),
         yaxt = "n", ...)
    axis(2, at = c(0, 1), labels = c("TO", "HS"), las = 1)

    # HS markers
    if (nrow(hs) > 0) {
      left_hs <- hs[hs$side == "left", ]
      right_hs <- hs[hs$side == "right", ]
      if (nrow(left_hs) > 0)
        points(left_hs$time, rep(1, nrow(left_hs)), pch = 25, col = "#2166ac",
               bg = "#2166ac", cex = 1.5)
      if (nrow(right_hs) > 0)
        points(right_hs$time, rep(1, nrow(right_hs)), pch = 25, col = "#b2182b",
               bg = "#b2182b", cex = 1.5)
    }

    # TO markers
    if (nrow(to) > 0) {
      left_to <- to[to$side == "left", ]
      right_to <- to[to$side == "right", ]
      if (nrow(left_to) > 0)
        points(left_to$time, rep(0, nrow(left_to)), pch = 24, col = "#2166ac",
               bg = "#2166ac", cex = 1.5)
      if (nrow(right_to) > 0)
        points(right_to$time, rep(0, nrow(right_to)), pch = 24, col = "#b2182b",
               bg = "#b2182b", cex = 1.5)
    }

    legend("topright",
           legend = c("Left", "Right"),
           col = c("#2166ac", "#b2182b"),
           pch = 15, cex = 0.9)

  } else if (type == "cycles") {
    cycles <- x$cycles
    if (!is.data.frame(cycles) || nrow(cycles) == 0) {
      message("No cycles to plot.")
      return(invisible(NULL))
    }
    stance_col <- if ("stance_pct" %in% names(cycles)) "stance_pct" else
      if ("stance_percentage" %in% names(cycles)) "stance_percentage" else NULL
    swing_col <- if ("swing_pct" %in% names(cycles)) "swing_pct" else
      if ("swing_percentage" %in% names(cycles)) "swing_percentage" else NULL
    if (is.null(stance_col) || is.null(swing_col)) {
      stop("cycles data must include stance/swing percentage columns", call. = FALSE)
    }
    method_label <- x$method
    if (is.null(method_label) || length(method_label) != 1L) method_label <- "unknown"
    barplot(
      rbind(cycles[[stance_col]], cycles[[swing_col]]),
      beside = FALSE, col = c("#2166ac", "#fee08b"),
      names.arg = paste(
        if ("side" %in% names(cycles)) cycles$side else rep("cycle", nrow(cycles)),
        seq_len(nrow(cycles))
      ),
      main = paste("Gait Cycles -", as.character(method_label)),
      ylab = "% of cycle",
      legend.text = c("Stance", "Swing"),
      args.legend = list(x = "topright")
    )
  }
  invisible(x)
}
