import numpy as np
import torch

import pyro 
import pyro.distributions as dist
from pyro.infer import Trace_ELBO, JitTrace_ELBO, NUTS, MCMC

from tqdm import trange

from pydevil.model import model
from pydevil.guide import guide
from pydevil.utils import prepare_batch, compute_disperion_prior

from sklearn.metrics.pairwise import rbf_kernel

def run_SVDE(
    input_matrix,
    model_matrix, 
    ncounts, 
    group_matrix = None,
    gene_specific_model_tensor = None,
    kernel_input = None,
    gene_names = None,
    cell_names = None,
    optimizer_name = "ClippedAdam",
    steps = 100, 
    lr = 0.1,
    gamma_lr = 0.1,
    cuda = False,
    jit_compile = False,
    batch_size = 5120, 
    full_cov = True, 
    prior_loc = 0.1,
    theta_bounds = (0., 1e16),
    init_loc = 0.1,
    threshold = 1e-5,
):
  
    if cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type(t=torch.FloatTensor)
        
    if batch_size > input_matrix.shape[0]:
        batch_size = input_matrix.shape[0]

    if kernel_input is not None:
        kernel_input = rbf_kernel(kernel_input, gamma = 1.) + np.eye(kernel_input.shape[0]) * 0.1
        kernel_input = torch.tensor(kernel_input).float()
        
    lrd = gamma_lr ** (1 / steps)
    
    input_matrix, model_matrix, UMI = torch.tensor(input_matrix).int(), torch.tensor(model_matrix).float(), torch.tensor(ncounts).float()

    if group_matrix is not None:
        group_matrix = torch.tensor(group_matrix).float()

    if gene_specific_model_tensor is not None:
        gene_specific_model_tensor = torch.tensor(gene_specific_model_tensor).float()
  
    if optimizer_name == "SGD":
        optimizer = pyro.optim.SGD({"lr": lr})
    elif optimizer_name == "Adam":
        optimizer = pyro.optim.Adam({"lr": lr})
    else:
        optimizer = pyro.optim.ClippedAdam({"lr": lr, "lrd" : lrd})
            
    elbo_list = [] 

    pyro.clear_param_store()
        
    svi = pyro.infer.SVI(model, guide, optimizer, pyro.infer.TraceGraph_ELBO())

    dispersion_priors, dispersion_var = compute_disperion_prior(X = input_matrix)
    
    if (threshold <= 0) :
      t = trange(steps, desc='Bar desc', leave = True)
      for _ in t:
  
          input_matrix_batch, model_matrix_batch, UMI_batch, group_matrix_batch, \
          gene_specific_model_tensor_batch, kernel_input_batch =  \
          prepare_batch(input_matrix, model_matrix, UMI, group_matrix,  \
          gene_specific_model_tensor, kernel_input, batch_size = batch_size)
  
          loss = svi.step(input_matrix_batch, model_matrix_batch, UMI_batch, dispersion_priors, dispersion_var, group_matrix_batch,
          gene_specific_model_tensor_batch, kernel_input_batch, full_cov = full_cov, 
          prior_loc = prior_loc,
          theta_bounds = theta_bounds,
          init_loc = init_loc)       
  
          elbo_list.append(loss)
          norm_ind = input_matrix_batch.shape[0] * input_matrix_batch.shape[1]
          t.set_description('ELBO: {:.5f}  '.format(loss / norm_ind))
          t.refresh()
    else:
        t = trange(steps, desc='Bar desc', leave = True)
        k_windows = int(steps * .01)
        previous_elbo = np.array([None for _ in range(k_windows)])
        converged_iterations = 0

        for i in t:
            input_matrix_batch, model_matrix_batch, UMI_batch, group_matrix_batch, \
            gene_specific_model_tensor_batch, kernel_input_batch =  \
            prepare_batch(input_matrix, model_matrix, UMI, group_matrix,  \
            gene_specific_model_tensor, kernel_input, batch_size = batch_size)
    
            loss = svi.step(input_matrix_batch, model_matrix_batch, UMI_batch, dispersion_priors, dispersion_var, group_matrix_batch,
            gene_specific_model_tensor_batch, kernel_input_batch, full_cov = full_cov, 
            prior_loc = prior_loc,
            theta_bounds = theta_bounds,
            init_loc = init_loc)       
    
            elbo_list.append(loss)
            norm_ind = input_matrix_batch.shape[0] * input_matrix_batch.shape[1]
            t.set_description('ELBO: {:.5f}  '.format(loss / norm_ind))
            t.refresh()

            previous_elbo[i % k_windows] = loss
            if i >= k_windows:
                moving_avg_elbo = np.mean(previous_elbo)
                moving_sd_elbo = np.std(previous_elbo)
                
                if (moving_sd_elbo / moving_avg_elbo) <= threshold:
                    converged_iterations += 1
                else:
                    converged_iterations = 0

                if converged_iterations == k_windows:
                    break
    
    n_features = model_matrix.shape[1]
    coeff = pyro.param("beta_mean")
    overdispersion = pyro.param("theta_p")

    if full_cov and n_features > 1:
        loc = torch.bmm(pyro.param("beta_loc"),pyro.param("beta_loc").permute(0,2,1))
    else:
        loc = pyro.param("beta_loc")

    eta = torch.exp(torch.matmul(model_matrix, coeff) + torch.unsqueeze(torch.log(UMI), 1) )
    lk = dist.NegativeBinomial(logits = eta - torch.log(overdispersion) ,
        total_count= torch.clamp(overdispersion, 1e-9,1e9)).log_prob(input_matrix).sum(dim = 0)

    if cuda and torch.cuda.is_available(): 
        input_matrix = input_matrix.cpu().detach().numpy() 
        overdispersion = overdispersion.cpu().detach().numpy() 
        eta = eta.cpu().detach().numpy()
        coeff = coeff.cpu().detach().numpy()
        loc = loc.cpu().detach().numpy()
        lk = lk.cpu().detach().numpy()
    else:
        input_matrix = input_matrix.detach().numpy() 
        overdispersion = overdispersion.detach().numpy() 
        eta = eta.detach().numpy()
        coeff = coeff.detach().numpy()
        loc = loc.detach().numpy()
        lk = lk.detach().numpy()

    #variance =  eta + eta**2 / overdispersion
    variance =  eta + eta**2 * overdispersion

    ret = {"loss" : elbo_list, "params" : {
        "theta" : overdispersion,
        "lk" : lk,
        "beta" : coeff,
        "eta" : eta,
        "variance" : loc
        }, "residuals" : (input_matrix - eta) / np.sqrt(variance),
        "hyperparams" : {"gene_names" : gene_names, "cell_names" : cell_names ,"model_matrix" : model_matrix}     }

    if group_matrix is not None:
        n_random = group_matrix.shape[1]
        if full_cov and n_random > 1:
            ret["params"]["random_effects_variance"] = torch.bmm(pyro.param("zeta_loc"),pyro.param("zeta_loc").permute(0,2,1))
        else:
            ret["params"]["random_effects_variance"] = pyro.param("zeta_loc")
        if cuda and torch.cuda.is_available():
            ret["params"]["random_effects_variance"] = ret["params"]["random_effects_variance"].cpu().detach().numpy()
        else:
            ret["params"]["random_effects_variance"] = ret["params"]["random_effects_variance"].detach().numpy()
            
    if kernel_input is not None:
        if cuda and torch.cuda.is_available():
            ret["params"]["lengthscale_kernel"] = pyro.param("lengthscale_param").cpu().detach().numpy()
        else:
            ret["params"]["lengthscale_kernel"] = pyro.param("lengthscale_param").detach().numpy()
        
    return ret
  
  
def run_HMC( input_matrix,
    model_matrix, 
    ncounts, 
    group_matrix = None,
    gene_specific_model_tensor = None,
    kernel_input = None,
    gene_names = None,
    cell_names = None,
    num_samples = 1000,
    num_chains = 4,
    warmup_steps = 200,
    cuda = False, 
    jit_compile = True,
    full_cov = True, 
    prior_loc = 0.1,
    theta_bounds = (0., 1e16),
    init_loc = 0.1):
              

    if cuda and torch.cuda.is_available():
        mp_context = "spawn"
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        mp_context = None
        torch.set_default_tensor_type(t=torch.FloatTensor)
  
    if jit_compile and not cuda:
        loss = JitTrace_ELBO
    else:
        loss = Trace_ELBO
    
    if jit_compile and not cuda:
        pyro_loss = JitTrace_ELBO
    else:
        pyro_loss = Trace_ELBO
        

    if kernel_input is not None:
        kernel_input = rbf_kernel(kernel_input, gamma = 1.) + np.eye(kernel_input.shape[0]) * 0.1
        kernel_input = torch.tensor(kernel_input).float()    
    
    input_matrix, model_matrix, UMI = torch.tensor(input_matrix).int(), torch.tensor(model_matrix).float(), torch.tensor(ncounts).float()

    if group_matrix is not None:
        group_matrix = torch.tensor(group_matrix).float()

    if gene_specific_model_tensor is not None:
        gene_specific_model_tensor = torch.tensor(gene_specific_model_tensor).float()
    
    input_matrix, model_matrix, UMI = torch.tensor(input_matrix).float(), torch.tensor(model_matrix).float(), torch.tensor(ncounts).float()

    nuts_kernel = NUTS(model)

    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains = num_chains,mp_context = mp_context)
    mcmc.run(input_matrix, model_matrix, UMI, group_matrix,
        gene_specific_model_tensor, kernel_input, full_cov = full_cov, 
        prior_loc = prior_loc,
        theta_bounds = theta_bounds,
        init_loc = init_loc)
    if cuda: 
        hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
    else:
        hmc_samples = {k: v.detach().numpy() for k, v in mcmc.get_samples().items()}
    ret = {"params" :  hmc_samples , 
             "hyperparams" : {"gene_names" : gene_names, "cell_names" : cell_names ,"model_matrix" : model_matrix}}

    return  ret
