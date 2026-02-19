\name{gk_detect}
\alias{gk_detect}
\title{Detect Gait Events}
\usage{
gk_detect(
  data,
  method = "bike",
  fps = NULL,
  angles = NULL,
  angles_align = "auto",
  require_core_markers = FALSE
)
}
\arguments{
  \item{data}{Input data. Either a named list that includes at least
  \code{angle_frames} or \code{fps} (for example from \code{gk_load_example()}),
  or a non-empty file path to a C3D file.}

  \item{method}{Non-empty character scalar method name (leading/trailing
  whitespace is ignored).
  Use \code{gk_methods()} to list options.}

  \item{fps}{Optional positive numeric sampling frequency in Hz.
  Required when it cannot be inferred from \code{data}.}

  \item{angles}{Optional path to external angles (MAT/CSV/JSON) used when
  \code{data} is a C3D file path.}

  \item{angles_align}{Alignment mode for external angles on the C3D timeline.
  Must be one of \code{"auto"}, \code{"second_hs"}, \code{"first_hs"},
  \code{"none"}, \code{"resample"}.}

  \item{require_core_markers}{Logical scalar. If \code{TRUE}, enforce
  core marker presence when proxy angles are computed from marker-only C3D.}
}
\value{
An object of class \code{gaitkit_result} containing:
\code{left_hs}, \code{right_hs}, \code{left_to}, \code{right_to},
\code{events}, \code{cycles}, \code{method}, \code{fps}, and \code{n_frames}.
}
\description{
Runs a single gait-event detector on motion capture input through the Python
\pkg{gaitkit} backend.
}
\examples{
\donttest{
trial <- gk_load_example("healthy")
result <- gk_detect(trial, method = "bike")
print(result)
}
}
