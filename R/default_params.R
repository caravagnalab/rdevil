SVI_default_args <- function() {
  list(
    optimizer_name = "ClippedAdam",
    steps = 500L,
    lr = 0.5,
    gamma_lr = 1e-04,
    cuda = FALSE,
    jit_compile = FALSE,
    batch_size = 5120L,
    full_cov = TRUE,
    gauss_loc = 5,
    theta_bounds = c(0, 1e5),
    disp_loc = .25
  )
}

HMC_default_agrs <- function() {
  list(
    num_samples = 1000,
    num_chains = 4,
    warmup_steps = 200,
    cuda = FALSE,
    jit_compile = TRUE,
    full_cov = TRUE,
    prior_loc = 10,
    theta_bounds = c(0., 1e16),
    init_loc = 10
  )
}
