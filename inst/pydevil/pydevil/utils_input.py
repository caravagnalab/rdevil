
import numpy as np
import torch
from pydevil.utils import estimate_size_factors, compute_offset_matrix, init_beta, compute_disperion_prior
import pyro
from sklearn.metrics.pairwise import rbf_kernel

def ensure_tensor(obj, cuda):
    """
    Ensure that the input object is converted into a PyTorch tensor.
    """
    if not isinstance(obj, torch.Tensor):
        obj = torch.tensor(obj)
    
    if cuda:
        if torch.cuda.is_available():
            obj = obj.cuda()
    return obj

def detach_tensor(obj):
    """
    Unload the tensor from the GPU.
    """
    if isinstance(obj, torch.Tensor):
        if obj.get_device() == 0:
            return obj.cpu().detach()
        else:
            return obj.detach()
    return obj
        
def detach_tensor_and_numpy(obj):
    """
    Unload the tensor from the GPU.
    """
    if isinstance(obj, torch.Tensor):
        if obj.get_device() == 0:
            return obj.cpu().detach().numpy()
        else:
            return obj.detach().numpy()
    return obj

def validate_boolean(parameter, parameter_name):
    """
    Validate whether the parameter is a boolean.
    """
    if not isinstance(parameter, bool):
        raise ValueError(f"{parameter_name} should be a boolean!")

def validate_integer(parameter, parameter_name):
    """
    Validate whether the parameter is an integer.
    """
    if not isinstance(parameter, int):
        raise ValueError(f"{parameter_name} should be an integer!")
    
def validate_numeric(parameter, parameter_name):
    """
    Validate whether the parameter is numeric.
    """
    if not isinstance(parameter, (int, (float, int, complex))):
        raise ValueError(f"{parameter_name} should be numeric!")

def validate_float(parameter, parameter_name):
    """
    Validate whether the parameter is a float.
    """
    if not isinstance(parameter, float):
        raise ValueError(f"{parameter_name} should be a float!")

def check_and_prepare_input_run_SVDE(input_matrix, model_matrix, size_factors, group_matrix,
                         gene_specific_model_tensor, kernel_input, gene_names,
                         cell_names, variance, optimizer_name, steps, lr, gamma_lr,
                         cuda, jit_compile, batch_size, full_cov, gauss_loc,
                         theta_bounds, disp_loc):
    """
    Check input parameters for SVDE model, validate their types, and compute additional quantities.
    """
    validate_boolean(cuda, "cuda")
    if cuda and torch.cuda.is_available():
       print("GPU usage was requested and and will be used!")
       torch.set_default_device("cuda:0")
    elif cuda and not torch.cuda.is_available():
       print("GPU usage was requested but is not available! Using CPU instead.")
       torch.set_default_device("cpu")
    elif not cuda and torch.cuda.is_available():
       print("GPU usage was not requested but is available! Using CPU instead.")
       torch.set_default_device("cpu")
    else:
       print("GPU usage was not requested and is not available! Using CPU.")
       torch.set_default_device("cpu")

    input_matrix = ensure_tensor(input_matrix, cuda=cuda).int()
    model_matrix = ensure_tensor(model_matrix, cuda=cuda).double()

    if input_matrix.size(0) != model_matrix.size(0):
        raise ValueError("The number of rows in the input_matrix and model_matrix should be the same!")
    
    print("Fitting SVDE model with", input_matrix.size(0), "cells and", input_matrix.size(1), "genes.")

    validate_boolean(size_factors, "size_factors")

    if group_matrix is not None:
        group_matrix = ensure_tensor(group_matrix, cuda=cuda).int()
        if input_matrix.size(0) != group_matrix.size(0):
            raise ValueError("The number of rows in the input_matrix and group_matrix should be the same!")
        
    if kernel_input is not None:
        kernel_input = rbf_kernel(kernel_input, gamma = 1.) + np.eye(kernel_input.shape[0]) * 0.1
        kernel_input = ensure_tensor(kernel_input, cuda=cuda).float()

    if gene_names is not None:
        if len(gene_names) != input_matrixinpu  .size(1):
            raise ValueError("The number of gene_names should be the same as the number of columns in model_matrix!")

    if cell_names is not None:
        if len(cell_names) != input_matrix.size(0):
            raise ValueError("The number of cell_names should be the same as the number of rows in input_matrix!")

    if variance not in ["VI_Estimate", "Hessian", "Sandwich"]:
        raise ValueError("Variance should be either 'VI_Estimate' or 'Hessian' or 'Sandwich'")

    if optimizer_name not in ["ClippedAdam", "Adam", "SGD"]:
        raise ValueError("optimizer_name should be either 'ClippedAdam' or 'Adam' or 'SGD'")

    validate_integer(steps, "steps")
    validate_float(lr, "lr")
    validate_float(gamma_lr, "gamma_lr")
    lrd = gamma_lr ** (1 / steps)

    validate_boolean(jit_compile, "jit_compile")

    validate_integer(batch_size, "batch_size")
    batch_size = min(batch_size, input_matrix.size(0))

    validate_boolean(full_cov, "full_cov")
    validate_numeric(gauss_loc, "gauss_loc")
    if not isinstance(theta_bounds, tuple):
        raise ValueError("theta_bounds should be a tuple!")
    validate_numeric(disp_loc, "disp_loc")

    # Produce new useful variables
    if size_factors:
        sf = estimate_size_factors(input_matrix, verbose=True)
        sf = torch.tensor(sf).double()
    else:
        sf = torch.ones(input_matrix.shape[0]).double()

    offset_matrix = compute_offset_matrix(input_matrix, sf)
    beta_estimate_matrix = init_beta(input_matrix, model_matrix, offset_matrix)

    if optimizer_name == "SGD":
        optimizer = pyro.optim.SGD({"lr": lr})
    elif optimizer_name == "Adam":
        optimizer = pyro.optim.Adam({"lr": lr})
    else:
        optimizer = pyro.optim.ClippedAdam({"lr": lr, "lrd": lrd})

    dispersion_priors = compute_disperion_prior(X=input_matrix.float(), offset_matrix=offset_matrix)
    dispersion_priors[dispersion_priors == 0] = min(dispersion_priors[dispersion_priors > 0])

    if group_matrix is not None:
        clusters = torch.tensor(group_matrix)
        group_matrix = None
    else:
        clusters = None

    # Return dictionary with input parameters and computed quantities
    return {
        'input_matrix': input_matrix,
        'model_matrix': model_matrix,
        'size_factors': size_factors,
        'sf': sf,
        'offset_matrix': offset_matrix,
        'group_matrix': group_matrix,
        'gene_specific_model_tensor': gene_specific_model_tensor,
        'kernel_input': kernel_input,
        'gene_names': gene_names,
        'cell_names': cell_names,
        'variance': variance,
        'optimizer_name': optimizer_name,
        'steps': steps,
        'lr': lr,
        'gamma_lr': gamma_lr,
        'lrd': lrd,
        'cuda': cuda,
        'jit_compile': jit_compile,
        'batch_size': batch_size,
        'full_cov': full_cov,
        'gauss_loc': gauss_loc,
        'theta_bounds': theta_bounds,
        'disp_loc': disp_loc,
        'beta_estimate_matrix': beta_estimate_matrix,
        'optimizer': optimizer,
        'dispersion_priors': dispersion_priors,
        'clusters': clusters
    }