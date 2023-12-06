import torch
import pyro 
import pyro.distributions as dist

def model(input_matrix, 
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
          theta_bounds = (1e-9, 1000000),
          disp_loc = 0.1):
    
    n_cells = input_matrix.shape[0]
    n_genes = input_matrix.shape[1]
    n_features = model_matrix.shape[1]

    # if group_matrix is not None:
    #     n_groups = group_matrix.shape[1]
    #     random_effects_loc = torch.zeros(n_groups)
    #     random_effects_scale = torch.eye(n_groups, n_groups) * .05
    #     random_effects = pyro.sample("random_effects", dist.MultivariateNormal(random_effects_loc, scale_tril=random_effects_scale))
    
    with pyro.plate("genes", n_genes, dim = -1): 
      #theta = pyro.sample("theta", dist.LogNormal(torch.log(dispersion_priors), torch.ones(n_genes) * disp_loc))
      theta = pyro.sample("theta", dist.Uniform(theta_bounds[0],theta_bounds[1]))
      # beta_prior_mu = beta_estimate.t()
      beta_prior_mu = torch.zeros(n_features)

      if full_cov:
        beta = pyro.sample("beta", dist.MultivariateNormal(beta_prior_mu, scale_tril=torch.eye(n_features, n_features) * gauss_loc, validate_args=False))
      else:
        beta = pyro.sample("beta", dist.Normal(beta_prior_mu, torch.ones(n_features) * gauss_loc).to_event(1))

      if group_matrix is not None:
        n_groups = group_matrix.shape[1]
        zeta = pyro.sample("zeta", dist.Normal(torch.zeros(n_genes, n_groups), torch.ones(n_groups) * gauss_loc / 100).to_event(1))

      with pyro.plate("data", n_cells, dim = -2):
        eta = torch.matmul(model_matrix, beta.T)  + torch.log(UMI).unsqueeze(1)
        
        # if group_matrix is not None:
        #     eta_zeta = torch.matmul(group_matrix, random_effects)
        #     eta = eta + eta_zeta.unsqueeze(1)

        if group_matrix is not None:
            eta_zeta = torch.matmul(group_matrix , zeta.T)
            eta_zeta = eta_zeta - eta_zeta.mean(dim=0)
            eta = eta + eta_zeta
        
        pyro.sample("obs", dist.NegativeBinomial(logits = eta - torch.log(1 / theta) , total_count=1/theta), obs = input_matrix)


def model_old(input_matrix, 
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
          theta_bounds = (1e-9, 1000000),
          disp_loc = 0.1):
  
  n_cells = input_matrix.shape[0]
  n_genes = input_matrix.shape[1]
  n_features = model_matrix.shape[1]
  lengthscale = torch.ones(n_genes)

  with pyro.plate("genes", n_genes, dim = -1):

    #theta = pyro.sample("theta", dist.Uniform(theta_bounds[0],theta_bounds[1]))
    #theta = torch.clamp(pyro.sample("theta", dist.Normal(dispersion_priors, init_loc)), theta_bounds[0], theta_bounds[1])

    theta = pyro.sample("theta", dist.LogNormal(torch.log(dispersion_priors), torch.ones(n_genes) * disp_loc))
    
    # theta = pyro.sample("theta", dist.LogNormal(dispersion_priors, dispersion_variance))

    beta_prior_mu = torch.zeros(n_features)

    if kernel_input is not None:
      lengthscale = pyro.sample("lengthscale", dist.Gamma(torch.ones(n_genes), torch.ones(n_genes)))
      kernel_mu = pyro.sample("kernel_random_effect", dist.MultivariateNormal(torch.zeros(n_cells), scale_tril=kernel_input, validate_args=False)).reshape([n_cells,n_genes]) * lengthscale

    
    if group_matrix is not None:
      n_groups = group_matrix.shape[1]
  ### This is one possible implementation as a hierarchical model ###
      if full_cov:
        zeta = pyro.sample("zeta", dist.MultivariateNormal(torch.zeros(n_genes, n_groups), scale_tril=torch.eye(n_groups, n_groups) * gauss_loc / 10, validate_args=False))
      else:
        zeta = pyro.sample("zeta", dist.Normal(torch.zeros(n_genes, n_groups), torch.ones(n_groups) * gauss_loc / 10).to_event(1))
        
    if gene_specific_model_tensor is not None:
        n_features_gs = gene_specific_model_tensor.shape[1]
        beta_prior_mu_gs = torch.zeros(n_features_gs)
        if full_cov:
            beta_gs = pyro.sample("beta_gs", dist.MultivariateNormal(beta_prior_mu_gs, scale_tril=torch.eye(n_features_gs, n_features_gs) * gauss_loc, validate_args=False))
        else:
            beta_gs = pyro.sample("beta_gs", dist.Normal(beta_prior_mu_gs, torch.ones(n_features_gs) * gauss_loc).to_event(1))
        
    if full_cov:
      beta = pyro.sample("beta", dist.MultivariateNormal(beta_prior_mu, scale_tril=torch.eye(n_features, n_features) * gauss_loc, validate_args=False))
    else:
      beta = pyro.sample("beta", dist.Normal(beta_prior_mu, torch.ones(n_features) * gauss_loc).to_event(1))

    with pyro.plate("data", n_cells, dim = -2):
        eta = torch.matmul(model_matrix, beta.T)  + torch.log(UMI).unsqueeze(1)
        
        if group_matrix is not None:
            eta_zeta = torch.matmul(group_matrix , zeta.T)
            eta = eta + eta_zeta
        if kernel_input is not None:
            eta = eta + kernel_mu
        if gene_specific_model_tensor is not None:
            eta = eta + eta_gene_specific

        # print(theta.max())
        # print(theta.min())
        #pyro.sample("obs", dist.GammaPoisson(rate = torch.clamp(torch.exp(eta + torch.log(1 / theta)),1e-9, 1e9) ,
        #concentration= torch.clamp(theta, 1e-9,1e9)), obs = input_matrix)
        
        # pyro.sample("obs", dist.GammaPoisson(concentration=theta, rate=theta / torch.exp(eta)), obs = input_matrix)
        
        pyro.sample("obs", dist.NegativeBinomial(logits = eta - torch.log(1 / theta) , total_count=1 / theta), obs = input_matrix)
    