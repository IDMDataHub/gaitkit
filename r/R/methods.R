#' Print a gaitkit_result
#'
#' @param x A \code{gaitkit_result} object.
#' @param ... Ignored.
#' @export
print.gaitkit_result <- function(x, ...) {
  n_hs <- length(x$left_hs) + length(x$right_hs)
  n_to <- length(x$left_to) + length(x$right_to)
  cat(sprintf("gaitkit_result  method=%s  fps=%.0f  frames=%d\n",
              x$method, x$fps, x$n_frames))
  cat(sprintf("  Heel-strikes: %d (L=%d, R=%d)\n",
              n_hs, length(x$left_hs), length(x$right_hs)))
  cat(sprintf("  Toe-offs:     %d (L=%d, R=%d)\n",
              n_to, length(x$left_to), length(x$right_to)))
  if (nrow(x$cycles) > 0) {
    st <- mean(x$cycles$stride_time, na.rm = TRUE)
    cad <- mean(x$cycles$cadence, na.rm = TRUE)
    cat(sprintf("  Stride time:  %.3f s (cadence %.0f steps/min)\n", st, cad))
  }
  invisible(x)
}

#' Summary of a gaitkit_result
#'
#' @param object A \code{gaitkit_result} object.
#' @param ... Ignored.
#' @export
summary.gaitkit_result <- function(object, ...) {
  print.gaitkit_result(object)
  if (nrow(object$cycles) > 0) {
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
  events <- x$events
  if (nrow(events) == 0) {
    message("No events to plot.")
    return(invisible(NULL))
  }

  if (type == "events") {
    hs <- events[events$event_type == "HS", ]
    to <- events[events$event_type == "TO", ]

    xlim <- range(events$time)
    ylim <- c(-0.5, 1.5)

    plot(xlim, ylim, type = "n", xlab = "Time (s)", ylab = "",
         main = paste("Gait Events \u2014", x$method), yaxt = "n", ...)
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
    if (nrow(cycles) == 0) {
      message("No cycles to plot.")
      return(invisible(NULL))
    }
    barplot(
      rbind(cycles$stance_pct, cycles$swing_pct),
      beside = FALSE, col = c("#2166ac", "#fee08b"),
      names.arg = paste(cycles$side, seq_len(nrow(cycles))),
      main = paste("Gait Cycles \u2014", x$method),
      ylab = "% of cycle",
      legend.text = c("Stance", "Swing"),
      args.legend = list(x = "topright")
    )
  }
  invisible(x)
}
