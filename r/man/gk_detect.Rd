\name{gk_detect}
\alias{gk_detect}
\title{Detect Gait Events}
\usage{
gk_detect(data, method = "bike", fps = NULL)
}
\arguments{
  \item{data}{Input data. Either a named list with \code{angle_frames} and
  \code{fps} (for example from \code{gk_load_example()}), or a path to a C3D file.}

  \item{method}{Character scalar method name. Use \code{gk_methods()} to list options.}

  \item{fps}{Optional numeric sampling frequency in Hz. Required when it cannot
  be inferred from \code{data}.}
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
\dontrun{
trial <- gk_load_example("healthy")
result <- gk_detect(trial, method = "bike")
print(result)
}
}
