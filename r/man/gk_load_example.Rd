\name{gk_load_example}
\alias{gk_load_example}
\title{Load a Bundled Example Trial}
\usage{
gk_load_example(name = "healthy")
}
\arguments{
  \item{name}{Example name. One of \code{"healthy"}, \code{"parkinson"},
  \code{"kuopio"}, \code{"stroke"}.}
}
\value{
A named list containing \code{angle_frames}, \code{fps}, \code{n_frames},
and metadata fields such as \code{description}, \code{source}, \code{doi},
and \code{population}.
}
\description{
Loads one of the gait examples bundled with the Python backend.
}
\examples{
\dontrun{
trial <- gk_load_example("healthy")
trial$fps
}
}
