#' Fit a generalized linear model for differential expression
#'
#' @description Fit a generalized linear model in order to
#' model the expression of genes coming from a scRNA count matrix.
#' The resulting object retains the inferred parameters that can
#' later be tested in order to find differentially expressed genes.
#'
#' @param input_matrix Matrix of counts representing gene expression
#' data for individual cells. Each row corresponds to a gene, and each
#' column represents a single cell.
#' @param model_matrix Matrix also known as design matrix, it represents
#' the relationship between the response variable and the predictor
#' variables in the model. Each row represents a cell and each columns
#' represent a predictor variable (e.g experimental conditions, biological
#' factors, treatment groups, batch effects, ...)
#' @param ncounts Vector with length equal to the number of cells. It
#' contains some kind of factor relate to the library size. For example,
#' it might contain the total UMI of each cell (i.e. colSums(input_matrix))
#' or the UMI of each cell divided by the average across all of them.
#' @param group_matrix .
#' @param gene_specific_model_tensor .
#' @param kernel_input .
#' @param gene_names Vector containing the names of the genes
#' @param cell_names Vector containing the names of the cells
#' @param inference_method String. Either "SVI" or "HMC"
#' @param method_specific_args List containing additional arguments.
#' The available arguments differs between the inference algorithms.
#'
#' SVI only:
#'
#' * `optimizer_name` optimizer, one of "ClippedAdam", "Adam", and "SGD";
#' * `steps` number of iterations of the optimization algorithm;
#' * `lr` learning rate for the optimize;
#' * `gamma_lr` parameters to tune the decay of the learning rate using "ClippedAdam";
#' * `batch_size` number of data points or observations sampled from the input matrix in
#' each iteration of the optimization algorithm;
#' * `threshold` ;
#' * `max_steps` ;
#'
#' HMC only:
#'
#' * `num_samples` number of iterations after the warmup-phase, it also indicates the
#' posterior samples each chain will produce;
#' * `num_chains` number of chains for the optimization algorithm;
#' * `warmup_steps` number of iterations of the warmup-phase;
#'
#' Shared:
#'
#' * `cuda` ;
#' * `jit_compile` ;
#' * `full_cov` ;
#' * `theta_bounds` ;
#' * `init_loc` ;
#'
#' @return A rdevil object of class `rdevil`
#' @export
fit_linear_model <- function(
    input_matrix,
    model_matrix,
    ncounts,
    group_matrix = NULL,
    gene_specific_model_tensor = NULL,
    kernel_input = NULL,
    gene_names = NULL,
    cell_names = NULL,
    inference_method = "SVI",
    method_specific_args = list()) {
  proc <- basilisk::basiliskStart(pydevil)

  ret <- basilisk::basiliskRun(proc,
    fun = function(
        input_matrix,
        model_matrix,
        ncounts,
        group_matrix,
        gene_specific_model_tensor,
        kernel_input,
        gene_names,
        cell_names,
        inference_method,
        method_specific_args) {
      py <- reticulate::import("pydevil")

      if (inference_method == "SVI") {
        if (length(method_specific_args) == 0) method_specific_args <- SVI_default_args()

        method_specific_args$input_matrix <- input_matrix
        method_specific_args$model_matrix <- model_matrix
        method_specific_args$ncounts <- ncounts
        method_specific_args$group_matrix <- group_matrix
        method_specific_args$gene_specific_model_tensor <- gene_specific_model_tensor
        method_specific_args$kernel_input <- kernel_input
        method_specific_args$gene_names <- gene_names
        method_specific_args$cell_names <- cell_names

        ret <- do.call(py$run_SVDE, method_specific_args)

        names(ret$params$theta) <- colnames(input_matrix)
        colnames(ret$params$eta) <- colnames(input_matrix)
        rownames(ret$params$eta) <- rownames(input_matrix)
        colnames(ret$residuals) <- colnames(input_matrix)
        rownames(ret$residuals) <- rownames(input_matrix)
        colnames(ret$params$beta) <- colnames(input_matrix)
      } else if (inference_method == "HMC") {
        if (length(method_specific_args) == 0) method_specific_args <- HMC_default_agrs()

        method_specific_args$input_matrix <- input_matrix
        method_specific_args$model_matrix <- model_matrix
        method_specific_args$ncounts <- ncounts
        method_specific_args$group_matrix <- group_matrix
        method_specific_args$gene_specific_model_tensor <- gene_specific_model_tensor
        method_specific_args$kernel_input <- kernel_input
        method_specific_args$gene_names <- gene_names
        method_specific_args$cell_names <- cell_names

        ret <- py$run_HMC(method_specific_args)

        dimnames(ret$params$beta) <- list(paste0("sample", 1:dim(ret$params$beta)[1]), colnames(input_matrix), rownames(model_matrix))
        dimnames(ret$params$theta) <- list(paste0("sample", 1:dim(ret$params$beta)[1]), colnames(input_matrix))
      } else if (inference_method == "MLE") {
        cli::cli_alert_danger("This inference method is not yet supported, have a look at the documentation for implemented algorithms or open an issue to request a new feature!")
      } else {
        cli::cli_alert_danger("This inference method is not yet supported, have a look at the documentation for implemented algorithms or open an issue to request a new feature!")
        stop()
      }

      ret$run_params <- method_specific_args
      ret$input_params$model_matrix <- model_matrix
      ret$input_params$ncounts <- ncounts

      class(ret) <- "rdevil"
      return(ret)
    },
    input_matrix = t(input_matrix),
    model_matrix = model_matrix,
    ncounts = ncounts,
    group_matrix = group_matrix,
    gene_specific_model_tensor = gene_specific_model_tensor,
    kernel_input = kernel_input,
    gene_names = gene_names,
    cell_names = cell_names,
    inference_method = inference_method,
    method_specific_args = method_specific_args
  )

  basilisk::basiliskStop(proc)

  ret
}
