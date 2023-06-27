import torch
import numpy as np
import pandas as pd
from functools import reduce


def check_is_categorical(mat):
    return (torch.sum(mat == 0) + torch.sum(mat == 1)) == (mat.shape[0] * mat.shape[1])

def init_beta(X, model_matrix, approx_type = "auto", nsamples = 10000):
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
