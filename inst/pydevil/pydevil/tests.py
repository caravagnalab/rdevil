import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd



def test_posterior_null(inference_res,contrast, alpha = 0.05):
    mu_test = (inference_res["params"]["beta"] * contrast.reshape([-1,1])).sum(axis = 0)
    
    variance_term = (inference_res["params"]["variance"].diagonal(0,2,1) * contrast**2).sum(axis = 1)

    # covariance_term = 0
    # if np.where(contrast > 0)[0] > 1:
    #     total_terms = len(contrast)
    #     for i in range(1, total_terms):
    #         for j in range(0, i - 1):
    #             covariance_term += contrast[i] * contrast[j] * inference_res["params"]["variance"][:,i,j]
    
    covariance_term = 0
    total_terms = len(contrast)
    for i in range(1, total_terms):
        for j in range(0, i - 1):
            covariance_term += contrast[i] * contrast[j] * inference_res["params"]["variance"][:,i,j]

    total_variance = variance_term + covariance_term

    p_value = 2 * (1 - norm.cdf(np.abs(mu_test), scale = np.sqrt(total_variance)))
    is_significant, p_value_adj = fdrcorrection(p_value, alpha=alpha)

    ret_df = pd.DataFrame.from_dict({
         "gene" : inference_res["hyperparams"]["gene_names"],
         "log_FC" : mu_test,
         "p_value" : p_value, 
         "p_value_adj" : p_value_adj, 
         "is_significant" : is_significant })
    return ret_df

def posterior_CI(inference_res,contrast, credible_mass = 0.95):
    mu_test = (inference_res["params"]["beta"] * contrast.reshape([-1,1])).sum(axis = 0)
    
    variance_term = (inference_res["params"]["variance"].diagonal(0,2,1) * contrast**2).sum(axis = 1)

    # covariance_term = 0
    # if np.where(contrast > 0)[0] > 1:
    #     total_terms = len(contrast)
    #     for i in range(1, total_terms):
    #         for j in range(0, i - 1):
    #             covariance_term += contrast[i] * contrast[j] * inference_res["params"]["variance"][:,i,j]

    covariance_term = 0
    total_terms = len(contrast)
    for i in range(1, total_terms):
        for j in range(0, i - 1):
            covariance_term += contrast[i] * contrast[j] * inference_res["params"]["variance"][:,i,j]

    total_variance = variance_term + covariance_term
    lower = (1 - credible_mass) / 2 
    upper = (1 + credible_mass) / 2     
    interval = np.array([norm.ppf(lower,loc = mu_test ,scale = np.sqrt(total_variance)),norm.ppf(upper,loc = mu_test ,scale = np.sqrt(total_variance))])
    ret_df = pd.DataFrame.from_dict({
         "gene" : inference_res["hyperparams"]["gene_names"],
         "log_FC" : mu_test,
         "CI_low" : interval[0,:], 
         "CI_high": interval[1,:], 
         "is_zero_in" : (interval[0,:] < 0) & (interval[1,:] > 0) })
    return ret_df

def test_posterior_ROPE(inference_res,contrast, LFC = 0.5):
    mu_test = (inference_res["params"]["beta"] * contrast.reshape([-1,1])).sum(axis = 0)
    
    variance_term = (inference_res["params"]["variance"].diagonal(0,2,1) * contrast**2).sum(axis = 1)

    # covariance_term = 0
    # if np.where(contrast > 0)[0] > 1:
    #     total_terms = len(contrast)
    #     for i in range(1, total_terms):
    #         for j in range(0, i - 1):
    #             covariance_term += contrast[i] * contrast[j] * inference_res["params"]["variance"][:,i,j]

    covariance_term = 0
    total_terms = len(contrast)
    for i in range(1, total_terms):
        for j in range(0, i - 1):
            covariance_term += contrast[i] * contrast[j] * inference_res["params"]["variance"][:,i,j]

    total_variance = variance_term + covariance_term

    ROPE = norm.cdf(LFC,loc = np.abs(mu_test), scale = np.sqrt(total_variance)) - norm.cdf(-LFC,loc = np.abs(mu_test), scale = np.sqrt(total_variance))

    ret_df = pd.DataFrame.from_dict({
         "gene" : inference_res["hyperparams"]["gene_names"],
         "log_FC" : mu_test,
         "ROPE" : ROPE
    })
    return ret_df


def test_posterior_null_HMC(inference_res,contrast, alpha = 0.05):
    beta_stacked = res_de_HMC["params"]["beta"]
    mu_test = (beta_stacked * contrast.reshape([1,1,-1])).sum(axis = 2)
    var_test = np.std(mu_test, axis = 0)
    mu_test = np.mean(mu_test, axis = 0)

    p_value = (1 - norm.cdf(np.abs(mu_test), scale = var_test))
    is_significant, p_value_adj = fdrcorrection(p_value, alpha=alpha)

    ret_df = pd.DataFrame.from_dict({
         "gene" : inference_res["hyperparams"]["gene_names"],
         "log_FC" : mu_test,
         "p_value" : p_value, 
         "p_value_adj" : p_value_adj, 
         "is_significant" : is_significant })
    return ret_df

def test_posterior_ROPE_HMC(inference_res,contrast, LFC = 0.5):
    
    beta_stacked = inference_res["params"]["beta"]
    mu_test = (beta_stacked * contrast.reshape([1,1,-1])).sum(axis = 2)
    ROPE = np.sum(np.abs(mu_test) < LFC,  axis = 0)/mu_test.shape[0]


    ret_df = pd.DataFrame.from_dict({
         "gene" : inference_res["hyperparams"]["gene_names"],
         "log_FC" : np.mean(mu_test, axis = 0),
         "ROPE" : ROPE 
    })
    return ret_df

def posterior_CI_HMC(inference_res,contrast, credible_mass = 0.95, CI_type = "HPDI"):
    beta_stacked = inference_res["params"]["beta"]
    mu_test = (beta_stacked * contrast.reshape([1,1,-1])).sum(axis = 2)
    if CI_type == "HPDI":
        interval = np.array([compute_hpdi(mu_test[:,i], credible_mass) for i in mu_test.shape[1]])
    else:
        interval = np.array([compute_quantile(mu_test[:,i], credible_mass) for i in mu_test.shape[1]])
    ret_df = pd.DataFrame.from_dict({
         "gene" : inference_res["hyperparams"]["gene_names"],
         "log_FC" : np.mean(mu_test, axis = 0),
         "CI_low" : interval[:,0], 
         "CI_high" : interval[:,1], 
         "is_zero_in" : (interval[:,0] < 0) & (interval[:,1] > 0) })
    
    
def compute_hpdi(samples, credible_mass=0.95):
    """
    Compute the HPDI of a given array of samples.

    Parameters
    ----------
    samples : numpy.ndarray
        A 1D array of samples.
    credible_mass : float, optional
        The desired mass of the HPDI interval (default is 0.95).

    Returns
    -------
    numpy.ndarray
        A 2-element array containing the lower and upper bounds of the HPDI.
    """
    sorted_samples = np.sort(samples)
    N = len(sorted_samples)
    n_samples_hpdi = int(np.floor(credible_mass * N))
    interval_widths = sorted_samples[n_samples_hpdi:] - sorted_samples[:-n_samples_hpdi]
    min_width_index = np.argmin(interval_widths)
    hpdi = np.array([sorted_samples[min_width_index],
                     sorted_samples[min_width_index + n_samples_hpdi]])

    return hpdi
    
    

def compute_quantile(samples, credible_mass=0.95):
    """
    Compute the quantile credible interval of a given array of samples.

    Parameters
    ----------
    samples : numpy.ndarray
        A 1D array of samples.
    credible_mass : float, optional
        The desired mass of the credible interval (default is 0.95).

    Returns
    -------
    numpy.ndarray
        A 2-element array containing the lower and upper bounds of the credible interval.
    """
    lower = (1 - credible_mass) / 2 * 100
    upper = (1 + credible_mass) / 2 * 100
    ci = np.percentile(samples, [lower, upper])

    return ci
   
