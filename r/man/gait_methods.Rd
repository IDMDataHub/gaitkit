\name{gait_methods}
\alias{gait_methods}
\title{List Available Detection Methods}
\usage{
gait_methods(module = "BIKEgait")
}
\arguments{
  \item{module}{Python module name. Defaults to \code{"BIKEgait"}.}
}
\value{Character vector of method names.}
\description{List methods exposed by the BIKEgait backend.}
\examples{
\dontrun{
library(BIKEgait)
gait_methods()
}
}
