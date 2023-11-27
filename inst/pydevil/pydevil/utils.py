import torch
import numpy as np
import pandas as pd
from functools import reduce
from scipy.optimize import curve_fit

def check_is_categorical(mat):
    return (torch.sum(mat == 0) + torch.sum(mat == 1)) == (mat.shape[0] * mat.shape[1])

def init_beta(Y, model_matrix, offset_matrix = None, approx_type = "auto", nsamples = 10000):
    if approx_type == "auto":
        if check_is_categorical(model_matrix):
            approx_type = "group_mean"
        else:
            approx_type = "batch_linear_regression"
    if approx_type == "group_mean":
        groups = torch.sum(model_matrix, axis = 1)
        norm_Y = Y / torch.exp(offset_matrix)
        unique_groups = torch.unique(groups)

        log_col_means_list = []

        for gr in unique_groups:
            mask = (groups == gr).view(-1, 1)
            col_means = torch.mean(norm_Y * mask, dim=0)
            log_col_means_list.append(torch.log(col_means))

        beta_init = torch.stack(log_col_means_list, dim=0)
        beta_init[beta_init == torch.tensor(-float("Inf"))] = 1e-9

    else:
        if nsamples < model_matrix.shape[0]:
            n_categorical = torch.where(torch.tensor([check_is_categorical(model_matrix[:,i].unsqueeze(1)) for i in range(model_matrix.shape[1])]))[0]
            if n_categorical.shape[0] == 0:
                indexes = np.random.randint(model_matrix.shape[0], size = nsamples)
            else:
                samples_per_cat = nsamples//n_categorical.shape[0]                
                n_categorical = n_categorical.tolist()
                indexes = []
                for i in n_categorical:
                    samples_i = samples_per_cat
                    if samples_per_cat > (model_matrix[:,i] == 1).sum().item():
                        samples_i = (model_matrix[:,i] == 1).sum().item()
                    av_samples = np.array(np.random.choice(torch.where(model_matrix[:,i] == 1)[0].detach().numpy(),size = samples_i, replace=False))
                    indexes.append(av_samples)
                indexes = np.concatenate(indexes)
            X_regression = Y[indexes,:]
            model_matrix_sub = model_matrix[indexes,:]
        else:
            X_regression = Y
            model_matrix_sub = model_matrix

        beta_init =  torch.linalg.solve(model_matrix_sub.t() @ model_matrix_sub, model_matrix_sub.t() @ X_regression.double(),  left = True)
    
    return beta_init

def init_beta_old(X, model_matrix, approx_type = "auto", nsamples = 10000):
    if approx_type == "auto":
        if check_is_categorical(model_matrix):
            approx_type = "group_mean"
        else:
            approx_type = "batch_linear_regression"
    if approx_type == "group_mean":
        beta_init = ((X.t() @ model_matrix) / model_matrix.sum(0)).t()
    else:
        if nsamples < model_matrix.shape[0]:
            n_categorical = torch.where(torch.tensor([check_is_categorical(model_matrix[:,i].unsqueeze(1)) for i in range(model_matrix.shape[1])]))[0]
            if n_categorical.shape[0] == 0:
                indexes = np.random.randint(model_matrix.shape[0], size = nsamples)
            else:
                samples_per_cat = nsamples//n_categorical.shape[0]                
                n_categorical = n_categorical.tolist()
                indexes = []
                for i in n_categorical:
                    samples_i = samples_per_cat
                    if samples_per_cat > (model_matrix[:,i] == 1).sum().item():
                        samples_i = (model_matrix[:,i] == 1).sum().item()
                    av_samples = np.array(np.random.choice(torch.where(model_matrix[:,i] == 1)[0].detach().numpy(),size = samples_i, replace=False))
                    indexes.append(av_samples)
                indexes = np.concatenate(indexes)
            X_regression = X[indexes,:]
            model_matrix_sub = model_matrix[indexes,:]
        else:
            X_regression = X
            model_matrix_sub = model_matrix
        beta_init =  torch.linalg.solve(model_matrix_sub.t() @ model_matrix_sub, model_matrix_sub.t() @ X_regression,  left = True)
    return beta_init

def init_theta(X, min_t = 0.0001, max_t = 1e6):
    variance = torch.var(X, dim=0)
    mean = torch.mean(X, dim=0)
    to_return = mean**2 / (variance - mean) 
    to_return = torch.clamp(torch.tensor(min_t), torch.tensor(max_t),to_return)
    to_return[torch.isnan(to_return)] = max_t
    return to_return.abs()

def rename_for_volcano(df_markers):
    df_markers_new = df_markers.set_axis(["GENE", "EFFECTSIZE", "p_value", "P", "is_significant"], axis=1)
    df_markers_new["SNP"] = df_markers_new["GENE"]
    df_markers_new["P"] = df_markers_new["P"] + 1e-16
    return df_markers_new

def prepare_batch(input_matrix, model_matrix, UMI, group_matrix, gene_specific_model_tensor, kernel_input, batch_size = 5124):
    
    if batch_size >= model_matrix.shape[0]:
        return input_matrix, model_matrix, UMI, group_matrix, gene_specific_model_tensor, kernel_input
    
    idx = torch.randperm(model_matrix.shape[0])[:batch_size]
    
    input_matrix_batch = input_matrix[idx,:]
    model_matrix_batch = model_matrix[idx,:]
    UMI_batch =  UMI[idx]
    
    if group_matrix is not None:
        group_matrix_batch = group_matrix[idx,:]
    else:
        group_matrix_batch = group_matrix
    
    if gene_specific_model_tensor is not None:
        gene_specific_model_tensor_batch = gene_specific_model_tensor[idx,:,:]
    else:
        gene_specific_model_tensor_batch = gene_specific_model_tensor
    
    if kernel_input is not None:
        kernel_input_batch = kernel_input[idx,:]
        kernel_input_batch = kernel_input_batch[:,idx]
    else:
        kernel_input_batch = kernel_input
    
    return input_matrix_batch, model_matrix_batch, UMI_batch, group_matrix_batch, gene_specific_model_tensor_batch, kernel_input_batch


def from_CNA_and_clones_to_mmatrix(input_CNA, cells_to_clones, gene_coord):
    
    clones = input_CNA.keys()
    clonal_profiles = pd.concat([build_CNA_profile_aux(gene_coord, input_CNA, clone) for clone in clones])
    common_genes = reduce(np.intersect1d,[cp.index.values for cp in clonal_profiles])
    clonal_profiles = [cp.loc[common_genes.tolist()] for cp in clonal_profiles]
    clonal_profiles = pd.concat(clonal_profiles, axis = 1)
    clonal_profiles = clonal_profiles.T.rename(columns= {"gene" : "clone"} )
    clonal_profiles["clone"] = clonal_profiles.index
    df_return = clonal_profiles.join(cell_to_clone.set_index('clone'), on = "clone").drop(columns=['clone', 'cell'])
    return df_return


def build_CNA_profile_aux(genes_to_chr,clones_profile, clone):
    cnv_df = clones_profile[clone]
    # Initialize an empty dataframe to store the result
    result_df = pd.DataFrame()

    # Loop over each row (gene) in the genes dataframe
    for gene_index, gene_row in genes_to_chr.iterrows():
        # Extract the start and end position of the current gene
        gene_start = gene_row['from']
        gene_end = gene_row['to']
        gene_chr = gene_row['chr']

        # Select rows (segments) from the cnv dataframe where the segment overlaps with the current gene
        # A segment overlaps with the gene if the segment starts before the gene ends AND the segment ends after the gene starts
        overlap_df = cnv_df[(cnv_df['start'] <= gene_end) & (cnv_df['end'] >= gene_start) & (cnv_df['chr'] == gene_chr)]

        # If any overlapping segments were found, add them to the result dataframe
        if not overlap_df.empty:
            overlap_df = overlap_df.copy()  # To avoid SettingWithCopyWarning
            overlap_df['gene'] = gene_row['gene']
            result_df = pd.concat([result_df, overlap_df])
    print(result_df)   
    return result_df.loc[:,"integer_copy_number"].set_index(result_df["gene"]).rename(columns = {"integer_copy_number" : clone})

def check_convergence(curr_p, previous_p, perc):
    percentage_diff = torch.abs(curr_p - previous_p) / abs(curr_p)
    return torch.sum(percentage_diff <= perc) >= (.95 * percentage_diff.numel())

def compute_disperion_prior(X, offset_matrix):
    def estimate_dispersions_by_moment(Y):
        xim = 1 / torch.mean(torch.mean(torch.exp(offset_matrix), dim=1))
        bv = torch.var(Y, dim=0)
        bm = torch.mean(Y, dim=0)
        return (bv - xim * bm) / (bm ** 2)

    def estimate_dispersions_roughly(Y):
        moments_disp = estimate_dispersions_by_moment(Y)
        disp_rough = torch.where(torch.isnan(moments_disp) | (moments_disp < 0), torch.tensor(0.0), moments_disp)
        return disp_rough
    
    return estimate_dispersions_roughly(X)
    # # Compute logarithm of geometric means
    # log_geo_means = torch.log(X).mean(dim=0, keepdim=True)

    # def calculate_sf(cnts):
    #     log_cnts = torch.log(cnts)
    #     valid_indices = torch.isfinite(log_geo_means) & (cnts > 0)
    #     filtered_log_diff = (log_cnts - log_geo_means)[valid_indices]
    #     return torch.exp(filtered_log_diff.median())

    # sf = torch.stack([calculate_sf(cnts) for cnts in X])

    # # Handle all-zero columns
    # all_zero_column = torch.isnan(sf) | (sf <= 0)
    # sf[all_zero_column] = float("nan")

    # if all_zero_column.any():
    #     num_zero_columns = all_zero_column.sum().item()
    #     print(f"{num_zero_columns} columns contain too many zeros to calculate a size factor. The size factor will be fixed to .0001.")
    #     sf = sf / torch.exp(torch.log(sf).mean())
    #     # sf[all_zero_column] = (torch.sum(X.float(),dim=1) / (torch.sum(X.float(),dim=1).mean(dim=0)))[all_zero_column]
    #     sf[all_zero_column] = 1000
    # else:
    #     sf = sf / torch.exp(torch.log(sf).mean())
        
    # # Compute rough dispersion
    # xim = 1 / sf.mean()
    # bv = X.float().var(dim=0)
    # bm = X.float().mean(dim=0)
    # dispersion_estimate = (bv - xim * bm) / bm.pow(2)

    # # Compute mean gene expression
    # mean_gene_expression = (X / sf[:, None]).mean(dim=0)

    # # Fit non-linear least squares regression
    # x = mean_gene_expression.cpu().detach().numpy()
    # y = dispersion_estimate.cpu().detach().numpy()
    # def trend(x, a0, a1):
    #     return a0 / x**(1/2) + a1

    # try:
    #     fit_params, _ = curve_fit(trend, x, y, check_finite=False, bounds=((0, 0), (1e16, 1e16)))
    # except:
    #     return 1 / dispersion_estimate, torch.tensor(.25)
    
    # fitted_a0 = fit_params[0]
    # fitted_a1 = fit_params[1]

    # # Estimate variance
    # dispersion_priors = torch.tensor(trend(mean_gene_expression.cpu().detach().numpy(), fitted_a0, fitted_a1))

    # # Calculate dispersion residuals
    # dispersion_residual = torch.log(dispersion_estimate) - torch.log(dispersion_priors)

    # # Calculate var_log_disp_est and exp_var_log_disp
    # var_log_disp_est = torch.median(torch.abs(dispersion_residual - torch.median(dispersion_residual))) * 1.4826
    # #exp_var_log_disp = trigamma((m - p) / 2)
    # if torch.isnan(var_log_disp_est): return dispersion_priors, torch.tensor(0.25)
    
    # dispersion_var = torch.max(var_log_disp_est, torch.tensor(0.25))

    return 1 / dispersion_priors, dispersion_var

def estimate_size_factors(X, verbose = False):
    if X.shape[1] <= 1:
        if verbose:
            print("Skipping size factor estimation! Only one gene is present!")
        return torch.ones(X.shape[0])

    sf = torch.sum(X, dim=1)

    # Check for all-zero columns
    all_zero_column = torch.isnan(sf) | (sf <= 0)

    # Replace all-zero columns with NA
    sf[all_zero_column] = 0

    if any(all_zero_column):
        warning_message = f"{torch.sum(all_zero_column)} columns contain too many zeros to calculate a size factor. The size factor will be fixed to 0.001"
        print(warning_message)
        
        # Apply the required transformations
        sf = sf / torch.exp(torch.mean(torch.log(sf[~all_zero_column]), dim=0, keepdim=True))
        sf[all_zero_column] = 0.001
    else:
        sf = sf / torch.exp(torch.mean(torch.log(sf), dim=0, keepdim=True))

    return sf

def compute_offset_matrix(Y, size_factors):
    n_samples, n_genes = Y.shape
    
    # Create the offset matrix
    offset_matrix = torch.tensor(0.0).expand((n_samples, n_genes))
    lsf = torch.log(size_factors)
    
    # Update the offset_matrix with size_factors
    offset_matrix = offset_matrix + lsf.view(-1,1)

    # Return the result
    return offset_matrix