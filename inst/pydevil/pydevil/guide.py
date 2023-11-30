import torch
import pyro 
import pyro.distributions as dist
from torch.distributions import constraints

def guide(input_matrix, 
          model_matrix, 
          UMI, 
          beta_estimate,
          dispersion_priors,
          group_matrix = None, 
          gene_specific_model_tensor = None,
          kernel_input = None,
          full_cov = True,
          gauss_loc = 10, 
          batch_size = 5120, 
          theta_bounds = (1e-6, 10000),
          disp_loc = 0.1):
  
    n_cells = input_matrix.shape[0]
    n_genes = input_matrix.shape[1]
    n_features = model_matrix.shape[1]
    
    # beta_estimate = init_beta(torch.log((input_matrix + 1e-5) / UMI.unsqueeze(1)), model_matrix)
    # theta_estimate = init_theta(input_matrix * 1.)
    theta_estimate = dispersion_priors
    beta_mean = pyro.param("beta_mean", beta_estimate, constraint=constraints.real)
            
    if kernel_input is not None:
        lengthscale_par = pyro.param("lengthscale_param", torch.ones(n_genes), constraint=constraints.positive)
        
    if group_matrix is not None:
        n_groups = group_matrix.shape[1]
        if full_cov:
            zeta_loc = pyro.param("zeta_loc", (torch.eye(n_groups, n_groups).repeat([n_genes,1,1]) * gauss_loc / 10), constraint=constraints.lower_cholesky)
        else:
            zeta_loc = pyro.param("zeta_loc", torch.ones(n_genes, n_groups) * gauss_loc / 10, constraint=constraints.positive) 
    
    with pyro.plate("genes", n_genes, dim = -1):
        if full_cov:
            beta_loc = pyro.param("beta_loc", (torch.eye(n_features, n_features).repeat([n_genes,1,1]) * gauss_loc), constraint=constraints.lower_cholesky)
        else:
            beta_loc = pyro.param("beta_loc", torch.ones(n_genes, n_features) * gauss_loc, constraint=constraints.positive)
    
        theta_p = pyro.param("theta_p", theta_estimate, constraint=constraints.positive)
        pyro.sample("theta", dist.Delta(theta_p))
        
        if kernel_input is not None:
            lengthscale = pyro.sample("lengthscale", dist.Delta(lengthscale_par))
            pyro.sample("kernel_random_effect", dist.MultivariateNormal(torch.zeros(n_cells), covariance_matrix= kernel_input, validate_args=False)).reshape([n_cells,n_genes]) * lengthscale
      
        if group_matrix is not None:
            if full_cov:
                zeta = pyro.sample("zeta", dist.MultivariateNormal(torch.zeros(n_genes, n_groups), scale_tril=zeta_loc, validate_args=False))
            else:
                zeta = pyro.sample("zeta", dist.Normal(torch.zeros(n_genes, n_groups), zeta_loc).to_event(1))

        if full_cov:
            pyro.sample("beta", dist.MultivariateNormal(beta_mean.t(),scale_tril = beta_loc, validate_args=False))
        else:
            pyro.sample("beta", dist.Normal(beta_mean.t(), beta_loc).to_event(1))


def guide_mle(*args, **kargs):
  pass
