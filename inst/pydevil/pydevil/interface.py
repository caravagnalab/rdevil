import numpy as np
import gc
import torch

import pyro 
from pyro.infer import Trace_ELBO, JitTrace_ELBO, NUTS, MCMC

from tqdm import trange

from pydevil.model import model
from pydevil.guide import guide
from pydevil.utils import prepare_batch
from pydevil.utils_input import check_and_prepare_input_run_SVDE, unload_tensor
from pydevil.utils_hessian import compute_hessians, compute_sandwiches

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
    input_data = check_and_prepare_input_run_SVDE(
        input_matrix, model_matrix, size_factors, group_matrix,
        gene_specific_model_tensor, kernel_input, 
        gene_names, cell_names, variance,
        optimizer_name, steps, lr, gamma_lr,
        cuda, jit_compile, batch_size, full_cov, 
        gauss_loc, theta_bounds, disp_loc
    )

    pyro.clear_param_store()
    svi = pyro.infer.SVI(model, guide, input_data['optimizer'], pyro.infer.TraceGraph_ELBO())

    elbo_list = []
    t = trange(steps, desc='Bar desc', leave = True)
    for it in t:
        input_matrix_batch, model_matrix_batch, UMI_batch, group_matrix_batch, \
        gene_specific_model_tensor_batch, kernel_input_batch = prepare_batch(
            input_data['input_matrix'], 
            input_data['model_matrix'], 
            input_data['sf'], 
            input_data['group_matrix'], 
            input_data['gene_specific_model_tensor'], 
            input_data['kernel_input'], 
            batch_size=batch_size
        )

        loss = svi.step(
            input_matrix_batch,
            model_matrix_batch,
            UMI_batch,
            input_data['beta_estimate_matrix'],
            input_data['dispersion_priors'],
            group_matrix_batch,
            gene_specific_model_tensor_batch,
            kernel_input_batch,
            full_cov=input_data['full_cov'],
            gauss_loc=input_data['gauss_loc'],
            theta_bounds=input_data['theta_bounds'],
            disp_loc=input_data['disp_loc']
        )

        elbo_list.append(loss)
        norm_ind = input_matrix_batch.shape[0] * input_matrix_batch.shape[1]
        t.set_description('ELBO: {:.5f}  '.format(loss / norm_ind))
        t.refresh()

    coeff = pyro.param("beta_mean").T
    overdispersion = pyro.param("theta_p")

    if input_data['variance'] == "VI_Estimate":
        n_features = input_data['model_matrix'].shape[1]
        if full_cov and n_features > 1:
            loc = torch.bmm(pyro.param("beta_loc"),pyro.param("beta_loc").permute(0,2,1))
        else:
            loc = pyro.param("beta_loc")
    elif input_data['variance'] == "Hessian":
        if input_data['clusters'] is None:
            loc = compute_hessians(
                input_matrix=input_data['input_matrix'], 
                model_matrix=input_data['model_matrix'], 
                coeff=coeff, 
                overdispersion=1 / overdispersion, 
                size_factors=input_data['sf'], 
                full_cov=input_data['full_cov'])
        else:
            loc = compute_sandwiches(
                input_matrix=input_data['input_matrix'], 
                model_matrix=input_data['model_matrix'], 
                coeff=coeff, 
                overdispersion=overdispersion, 
                size_factors=input_data['sf'], 
                cluster=input_data['clusters']
            )

    #eta = torch.exp(torch.matmul(model_matrix, coeff) + torch.unsqueeze(torch.log(UMI), 1) )
    #variance =  eta + eta**2 / overdispersion
    #lk = dist.NegativeBinomial(logits = eta - torch.log(overdispersion) ,
    #    total_count= torch.clamp(overdispersion, 1e-9,1e9)).log_prob(input_matrix).sum(dim = 0)
            
    input_matrix = unload_tensor(input_matrix)
    model_matrix = unload_tensor(model_matrix)
    overdispersion = unload_tensor(overdispersion)
    coeff = unload_tensor(coeff)
    loc = unload_tensor(loc)
    UMI = unload_tensor(input_data['sf'])

    # if cuda and torch.cuda.is_available():
    #     input_matrix = input_matrix.cpu().detach().numpy()
    #     model_matrix = model_matrix.cpu().detach().numpy()
    #     overdispersion = overdispersion.cpu().detach().numpy()
    #     #eta = eta.cpu().detach().numpy()
    #     # variance = variance.cpu().detach().numpy()
    #     coeff = coeff.cpu().detach().numpy()
    #     loc = loc.cpu().detach().numpy()
    #     # lk = lk.cpu().detach().numpy()
    #     UMI = input_data['sf'].cpu().detach().numpy()
    # else:
    #     input_matrix = input_matrix.detach().numpy()
    #     model_matrix = model_matrix.detach().numpy()
    #     overdispersion = overdispersion.detach().numpy()
    #     #eta = eta.detach().numpy()
    #     coeff = coeff.detach().numpy()
    #     #variance = variance.detach().numpy()
    #     loc = loc.detach().numpy()
    #     # lk = lk.detach().numpy()
    #     UMI = input_data['sf'].detach().numpy()

    ret = {
        "loss" : elbo_list,
        "params" : {
            "theta" : overdispersion,
            #"lk" : lk,
            "beta" : coeff,
            "eta" : eta,
            "variance" : loc,
            "size_factors" : UMI
        },
        #"residuals" : (input_matrix - eta) / np.sqrt(variance),
        "hyperparams" : {
            "gene_names" : gene_names,
            "cell_names" : cell_names ,
            "model_matrix" : model_matrix,
            "full_cov" : full_cov
        }
    }

    if cuda and torch.cuda.is_available():
        del elbo_list, overdispersion, coeff, loc, variance# , lk, eta
        del input_matrix, model_matrix, group_matrix, UMI, gene_specific_model_tensor, kernel_input
        del input_matrix_batch, model_matrix_batch, group_matrix_batch, UMI_batch, gene_specific_model_tensor_batch, kernel_input_batch
        del loss, svi
        del input_data

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
