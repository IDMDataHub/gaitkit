\name{gk_detect_ensemble}
\alias{gk_detect_ensemble}
\title{Ensemble Gait Event Detection}
\usage{
gk_detect_ensemble(data, methods = NULL, min_votes = 2L, tolerance_ms = 50, fps = NULL)
}
\arguments{
  \item{data}{Input data, same contract as \code{gk_detect()}.}

  \item{methods}{Optional character vector of methods to include in the vote.
  When provided, must contain at least two methods.}

  \item{min_votes}{Integer minimum number of agreeing methods for consensus
  (must be >= 1).}

  \item{tolerance_ms}{Numeric temporal clustering tolerance in milliseconds
  (must be >= 0).}

  \item{fps}{Optional positive numeric sampling frequency in Hz.}
}
\value{
A \code{gaitkit_result} object with consensus events and confidence scores.
}
\description{
Runs multiple detectors and combines their outputs with temporal clustering and
majority voting.
}
\examples{
\dontrun{
trial <- gk_load_example("healthy")
res <- gk_detect_ensemble(trial, methods = c("bike", "zeni", "oconnor"))
}
}
