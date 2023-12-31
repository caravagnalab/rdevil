% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tests.R
\name{test_posterior_null}
\alias{test_posterior_null}
\title{Test null posterior SVI}
\usage{
test_posterior_null(inference_res, contrast, alpha = 0.05)
}
\arguments{
\item{inference_res}{results coming from \code{fit_linear_model} function}

\item{contrast}{Vector with the same number of rows as the number of
conditions in the design matrix indicating how the coefficients
estimated should be tested against each other.}

\item{alpha}{Significance level for the p-values. Default is 0.05.}
}
\value{
A data.frame with \code{gene}, \code{log_FC}, \code{p_value},
\code{p_value_adj} and \code{is_significant} columns
}
\description{
Test null posterior SVI
}
