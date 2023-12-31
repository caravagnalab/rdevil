import numpy as np
import gc
import torch

import pyro 
import pyro.distributions as dist
from pyro.infer import Trace_ELBO, JitTrace_ELBO, NUTS, MCMC

from tqdm import trange

from pydevil.model import model
from pydevil.scheduler import myOneCycleLR
from pydevil.guide import guide
from pydevil.utils import prepare_batch, compute_disperion_prior, init_beta, compute_offset_matrix, estimate_size_factors
from pydevil.utils_hessian import compute_hessians, compute_sandwiches

from sklearn.metrics.pairwise import rbf_kernel

def run_SVDE(
    input_matrix,
    model_matrix, 
    size_factors = True,
    group_matrix = None,
    gene_specific_model_tensor = None,
    kernel_input = None,
    gene_names = None,
    cell_names = None,
    variance = "VI_Estimate",
    optimizer_name = "ClippedAdam",
    steps = 500, 
    lr = 0.5,
    gamma_lr = 1e-04,
    cuda = False,
    jit_compile = False,
    batch_size = 5120, 
    full_cov = True, 
    gauss_loc = 5,
    theta_bounds = (0., 1e16),
    disp_loc = .25
):
    torch.set_default_dtype(torch.float64)
    if cuda and torch.cuda.is_available():
        #torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        torch.set_default_device("cuda:0")
    else:
        torch.set_default_device("cpu")

    if variance not in ["VI_Estimate", "Hessian", "Sandwich"]:
        raise ValueError("Variance should be either 'VI_Estimate' or 'Hessian' or 'Sandwich'")
        
    if batch_size > input_matrix.shape[0]:
        batch_size = input_matrix.shape[0]

    if kernel_input is not None:
        kernel_input = rbf_kernel(kernel_input, gamma = 1.) + np.eye(kernel_input.shape[0]) * 0.1
        kernel_input = torch.tensor(kernel_input).float()
        
    lrd = gamma_lr ** (1 / steps)

    input_matrix, model_matrix = torch.tensor(input_matrix).int(), torch.tensor(model_matrix)
    if size_factors:
        sf = estimate_size_factors(input_matrix, verbose = True)
        UMI = torch.tensor(sf).float()
    else:
        sf = UMI = torch.ones(input_matrix.shape[0])

    offset_matrix = compute_offset_matrix(input_matrix, sf)
    # beta_estimate_matrix = init_beta(torch.log((input_matrix + 1e-5) / UMI.unsqueeze(1)), model_matrix)
    beta_estimate_matrix = init_beta(torch.tensor(input_matrix), model_matrix, offset_matrix)

    if group_matrix is not None:
        clusters = torch.tensor(group_matrix)
        group_matrix = None
    else:
        clusters = None

    if gene_specific_model_tensor is not None:
        gene_specific_model_tensor = torch.tensor(gene_specific_model_tensor, dtype = torch.float32)
  
    if optimizer_name == "SGD":
        optimizer = pyro.optim.SGD({"lr": lr})
    elif optimizer_name == "Adam":
        optimizer = pyro.optim.Adam({"lr": lr})
    elif optimizer_name == "OneCycleLR":
        optimizer = pyro.optim.PyroOptim(myOneCycleLR, {'lr': lr, "lrd" : gamma_lr, "steps": steps})
    else:
        optimizer = pyro.optim.ClippedAdam({"lr": lr, "lrd" : lrd})
            
    elbo_list = [] 
    beta_list = []
    overdisp_list = []

    pyro.clear_param_store()
        
    svi = pyro.infer.SVI(model, guide, optimizer, pyro.infer.TraceGraph_ELBO())

    dispersion_priors = compute_disperion_prior(X = input_matrix.float(), offset_matrix = offset_matrix)
    dispersion_priors[dispersion_priors == 0] = min(dispersion_priors[dispersion_priors > 0])

    t = trange(steps, desc='Bar desc', leave = True)
    for it in t:
        input_matrix_batch, model_matrix_batch, UMI_batch, group_matrix_batch, \
        gene_specific_model_tensor_batch, kernel_input_batch =  \
        prepare_batch(input_matrix, model_matrix, UMI, group_matrix,  \
        gene_specific_model_tensor, kernel_input, batch_size = batch_size)

        loss = svi.step(input_matrix_batch, model_matrix_batch, UMI_batch, beta_estimate_matrix, dispersion_priors, group_matrix_batch,
        gene_specific_model_tensor_batch, kernel_input_batch, full_cov = full_cov, gauss_loc = gauss_loc, theta_bounds = theta_bounds, disp_loc = disp_loc)  

        elbo_list.append(loss)
        norm_ind = input_matrix_batch.shape[0] * input_matrix_batch.shape[1]
        t.set_description('ELBO: {:.5f}  '.format(loss / norm_ind))
        t.refresh()
        
    coeff = pyro.param("beta_mean").T
    overdispersion = pyro.param("theta_p")

    if variance == "VI_Estimate":
        n_features = model_matrix.shape[1]
        if full_cov and n_features > 1:
            loc = torch.bmm(pyro.param("beta_loc"),pyro.param("beta_loc").permute(0,2,1))
        else:
            loc = pyro.param("beta_loc")
    elif variance == "Hessian":
        if clusters is None:
            loc = compute_hessians(input_matrix=input_matrix, model_matrix=model_matrix, coeff=coeff, overdispersion=1 / overdispersion, size_factors=UMI, full_cov=full_cov)
        else:
            loc = compute_sandwiches(input_matrix, model_matrix, coeff, overdispersion, UMI, clusters)

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
        UMI = UMI.cpu().detach().numpy()
    else:
        input_matrix = input_matrix.detach().numpy() 
        overdispersion = overdispersion.detach().numpy() 
        eta = eta.detach().numpy()
        coeff = coeff.detach().numpy()
        loc = loc.detach().numpy()
        lk = lk.detach().numpy()
        UMI = UMI.detach().numpy()

    variance =  eta + eta**2 / overdispersion
    # variance =  eta + eta**2 * overdispersion

    ret = {
        "loss" : elbo_list, 
        "params" : {
            "theta" : overdispersion,
            "lk" : lk,
            "beta" : coeff,
            "eta" : eta,
            "variance" : loc,
            "size_factors" : UMI
        }, 
        "residuals" : (input_matrix - eta) / np.sqrt(variance),
        "hyperparams" : {
            "gene_names" : gene_names, 
            "cell_names" : cell_names ,
            "model_matrix" : model_matrix,
            "full_cov" : full_cov
        }     
    }

    # if group_matrix is not None:
    #     ret['params']['random_effects'] = pyro.param("random_effects_loc").cpu().detach().numpy()

    #if group_matrix is not None:
        #ret['params']['subject_dispersion'] = pyro.param("s").cpu().detach().numpy()
        #ret['params']['random_effects'] = pyro.param("w").cpu().detach().numpy()
        #ret['params']['alpha'] = pyro.param("alpha_p").cpu().detach().numpy()
        #ret['params']['lambda'] = pyro.param("lambda_p").cpu().detach().numpy()
        #ret['params']['random_effects'] = pyro.param("random_effects_p").cpu().detach().numpy()

    # if group_matrix is not None:
    #     n_random = group_matrix.shape[1]
    #     if full_cov and n_random > 1:
    #         ret["params"]["random_effects_variance"] = torch.bmm(pyro.param("zeta_loc"),pyro.param("zeta_loc").permute(0,2,1))
    #     else:
    #         ret["params"]["random_effects_variance"] = pyro.param("zeta_loc")
    #     if cuda and torch.cuda.is_available():
    #         ret["params"]["random_effects_variance"] = ret["params"]["random_effects_variance"].cpu().detach().numpy()
    #     else:
    #         ret["params"]["random_effects_variance"] = ret["params"]["random_effects_variance"].detach().numpy()
            
    # if kernel_input is not None:
    #     if cuda and torch.cuda.is_available():
    #         ret["params"]["lengthscale_kernel"] = pyro.param("lengthscale_param").cpu().detach().numpy()
    #     else:
    #         ret["params"]["lengthscale_kernel"] = pyro.param("lengthscale_param").detach().numpy()

    if cuda and torch.cuda.is_available():
        del elbo_list, beta_list, overdisp_list
        del overdispersion, lk, coeff, eta, loc, variance
        del input_matrix, model_matrix, group_matrix, beta_estimate_matrix, UMI, gene_specific_model_tensor, kernel_input
        del input_matrix_batch, model_matrix_batch, group_matrix_batch, UMI_batch, gene_specific_model_tensor_batch, kernel_input_batch
        del loss
        torch.cuda.empty_cache()
        gc.collect()

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
