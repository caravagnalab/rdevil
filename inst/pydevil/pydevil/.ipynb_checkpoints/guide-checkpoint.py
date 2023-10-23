import pandas as pd
import numpy as np
import torch

import pyro 
import pyro.distributions as dist
from torch.distributions import constraints

from pydevil.utils import init_beta, init_theta


def guide(input_matrix, 
          model_matrix, 
          UMI, 
          dispersion_priors,
          dispersion_variance,
          group_matrix = None, 
          gene_specific_model_tensor = None,
          kernel_input = None,
          full_cov = True,
          prior_loc = 10, 
          batch_size = 5120, 
          theta_bounds = (1e-6, 10000),
          init_loc = 0.1):
  
    n_cells = input_matrix.shape[0]
    n_genes = input_matrix.shape[1]
    n_features = model_matrix.shape[1]
    
    # theta_estimate = init_theta(input_matrix * 1.)
    theta_estimate = dispersion_priors

        
def guide_mle(*args, **kargs):
  pass
