#' Test null posterior SVI
#'
#' @param inference_res results coming from `fit_linear_model` function
#' @param contrast Vector with the same number of rows as the number of
#' conditions in the design matrix indicating how the coefficients
#' estimated should be tested against each other.
#' @param alpha Significance level for the p-values. Default is 0.05.
#'
#' @return
#' @export
#'
#' @examples
test_posterior_null <- function(
    inference_res,
    contrast,
    alpha = 0.05) {
  proc <- basilisk::basiliskStart(pydevil)

  ret <- basilisk::basiliskRun(proc,
    fun = function(inference_res, contrast, alpha) {
      py <- reticulate::import("pydevil")

      arg_list <- list()
      arg_list$inference_res <- inference_res
      arg_list$contrast <- contrast
      arg_list$alpha <- alpha

      ret <- do.call(py$test_posterior_null, arg_list)

      return(ret)
    },
    inference_res = inference_res,
    contrast = contrast,
    alpha = alpha
  )

  basilisk::basiliskStop(proc)
  ret
}

#' Test credible interval SVI
#'
#' @param inference_res results coming from `fit_linear_model` function
#' @param contrast Vector with the same number of rows as the number of
#' conditions in the design matrix indicating how the coefficients
#' estimated should be tested against each other.
#' @param credible_mass
#'
#' @return
#' @export
#'
#' @examples
test_posterior_CI <- function(
    inference_res,
    contrast,
    credible_mass = 0.95) {
  proc <- basilisk::basiliskStart(pydevil)

  ret <- basilisk::basiliskRun(proc,
    fun = function(inference_res, contrast, credible_mass) {
      py <- reticulate::import("pydevil")

      arg_list <- list()
      arg_list$inference_res <- inference_res
      arg_list$contrast <- contrast
      arg_list$credible_mass <- credible_mass

      ret <- do.call(py$posterior_CI, arg_list)

      return(ret)
    },
    inference_res = inference_res,
    contrast = contrast,
    credible_mass = credible_mass
  )

  basilisk::basiliskStop(proc)
  ret
}

#' Test posterior ROPE SVI
#'
#' @param inference_res results coming from `fit_linear_model` function
#' @param contrast Vector with the same number of rows as the number of
#' conditions in the design matrix indicating how the coefficients
#' estimated should be tested against each other.
#' @param LFC
#'
#' @return
#' @export
#'
#' @examples
test_posterior_ROPE <- function(
    inference_res,
    contrast,
    LFC = 0.5) {
  proc <- basilisk::basiliskStart(pydevil)

  ret <- basilisk::basiliskRun(proc,
    fun = function(inference_res, contrast, LFC) {
      py <- reticulate::import("pydevil")

      arg_list <- list()
      arg_list$inference_res <- inference_res
      arg_list$contrast <- contrast
      arg_list$LFC <- LFC

      ret <- do.call(py$test_posterior_ROPE, arg_list)

      return(ret)
    },
    inference_res = inference_res,
    contrast = contrast,
    LFC = LFC
  )

  basilisk::basiliskStop(proc)
  ret
}
