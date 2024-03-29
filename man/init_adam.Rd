% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/adam.R
\name{init_adam}
\alias{init_adam}
\title{Initialize state for Adam optimization}
\usage{
init_adam(
  parameters,
  beta1 = 0.9,
  beta2 = 0.999,
  epsilon = 1e-08,
  learning_rate = 0.01
)
}
\arguments{
\item{parameters}{(Named) list of initial parameters}

\item{beta1}{Controls forgetting rate for E[g]}

\item{beta2}{Controls forgetting rate for E[g^2]}

\item{epsilon}{Prevents divide by 0}

\item{learning_rate}{I think you can guess this one.}
}
\description{
Initialize state for Adam optimization
}
