\name{gk_install_python}
\alias{gk_install_python}
\title{Install Python Backend}
\usage{
gk_install_python(envname = NULL, pip = TRUE)
}
\arguments{
  \item{envname}{Optional environment name passed to \code{reticulate::py_install()}.}

  \item{pip}{Logical flag, whether to use \code{pip}.}
}
\value{
Invisibly returns \code{TRUE} on success.
}
\description{
Convenience helper to install the Python \pkg{gaitkit} backend in the active
or selected reticulate environment.
}
\examples{
\dontrun{
gk_install_python()
}
}
