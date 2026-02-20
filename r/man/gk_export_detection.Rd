\name{gk_export_detection}
\alias{gk_export_detection}
\title{Export Detection Results}
\usage{
gk_export_detection(result, output_prefix, formats = "json")
}
\arguments{
  \item{result}{A \code{gaitkit_result} object from \code{gk_detect()} /
  \code{gk_detect_ensemble()}, or a structured payload list with
  \code{meta}, \code{heel_strikes}, \code{toe_offs}, and \code{cycles}.}

  \item{output_prefix}{Non-empty character scalar path prefix for output files.}

  \item{formats}{Character vector of output formats. Supported values:
  \code{"json"}, \code{"csv"}, \code{"xlsx"}, \code{"myogait"}.}
}
\value{
Named character vector of written file paths.
}
\description{
Exports detection results through the Python compatibility exporter.
This enables writing standard exports and a proprietary MyoGait-compatible
\code{events} JSON.
}
\examples{
\donttest{
trial <- gk_load_example("healthy")
res <- gk_detect(trial, method = "bike")
gk_export_detection(res, tempfile("gaitkit_out"), formats = c("json", "myogait"))
}
}
