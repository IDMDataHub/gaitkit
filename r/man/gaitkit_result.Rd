\name{gaitkit_result}
\alias{print.gaitkit_result}
\alias{summary.gaitkit_result}
\alias{plot.gaitkit_result}
\title{Methods for \code{gaitkit_result}}
\usage{
\method{print}{gaitkit_result}(x, ...)

\method{summary}{gaitkit_result}(object, ...)

\method{plot}{gaitkit_result}(x, type = "events", ...)
}
\arguments{
  \item{x}{A \code{gaitkit_result} object.}

  \item{object}{A \code{gaitkit_result} object.}

  \item{type}{Plot type: \code{"events"} (timeline) or \code{"cycles"}.}

  \item{...}{Additional arguments passed through.}
}
\value{
\code{print()} and \code{summary()} return the input object invisibly.
\code{plot()} returns the input object invisibly after plotting.
}
\description{
S3 methods for printing, summarizing and plotting outputs returned by
\code{gk_detect()} and \code{gk_detect_ensemble()}.

For event timelines, both event label styles are accepted:
\code{"HS"}/\code{"TO"} and \code{"heel_strike"}/\code{"toe_off"}.

The events plot expects \code{x$events} to include columns
\code{time}, \code{event_type}, and \code{side}.
}
\examples{
\donttest{
trial <- gk_load_example("healthy")
res <- gk_detect(trial)
print(res)
summary(res)
plot(res)
}
}
