\name{gait_detect}
\alias{gait_detect}
\title{Detect Gait Events From Structured Frames}
\usage{
gait_detect(method, frames, fps = 100, module = "BIKEgait", units = list(position = "mm", angles = "deg"))
}
\arguments{
  \item{method}{Detector method name.}
  \item{frames}{List of structured frame objects.}
  \item{fps}{Sampling frequency (Hz).}
  \item{module}{Python module name. Defaults to \code{"BIKEgait"}.}
  \item{units}{Named list with \code{position} (\code{"mm"}/\code{"m"}) and \code{angles} (\code{"deg"}/\code{"rad"}).}
}
\value{Named list with \code{meta}, \code{heel_strikes}, \code{toe_offs}, and \code{cycles}.}
\description{Bridge function calling \code{BIKEgait.detect_events_structured} via reticulate.}
\examples{
\dontrun{
library(BIKEgait)

frames <- list(
  list(
    frame_index = 0L,
    left_hip_angle = 5,
    right_hip_angle = 5,
    left_knee_angle = 10,
    right_knee_angle = 10,
    left_ankle_angle = 0,
    right_ankle_angle = 0,
    landmark_positions = list(
      left_knee = c(0, 0, 0),
      right_knee = c(0, 0, 0),
      left_ankle = c(0, 0, 0),
      right_ankle = c(0, 0, 0)
    )
  )
)

res <- gait_detect(
  method = "bayesian_bis",
  frames = frames,
  fps = 100,
  units = list(position = "mm", angles = "deg")
)

str(res$meta)
}
}
