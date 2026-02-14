\name{gait_install_python}
\alias{gait_install_python}
\title{Install Python Backend (BIKEgait)}
\usage{
gait_install_python(package = "BIKEgait", envname = NULL, pip = TRUE)
}
\arguments{
  \item{package}{Python package name.}
  \item{envname}{Optional reticulate environment name.}
  \item{pip}{Logical, whether to use pip.}
}
\value{Invisible \code{TRUE} on success.}
\description{Convenience helper to install the BIKEgait Python backend package from R.}
\examples{
\dontrun{
library(BIKEgait)
gait_install_python()
}
}
