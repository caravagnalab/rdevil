import torch
from tqdm import trange

def compute_hessian(obs, model_matrix, coeff, overdispersion):
    beta = torch.tensor(coeff)
    model_matrix = torch.tensor(model_matrix)
    alpha = 1 / overdispersion
    design_v = model_matrix.t()  # Transpose for vectorized operations
    yi = obs.unsqueeze(1)  # Add a new axis for broadcasting
    k = torch.exp(torch.matmul(design_v.t(), beta))
    gamma_sq = (1 + alpha * k) ** 2

    xij = torch.einsum('ik,jk->ijk', design_v, design_v).permute(2,0,1)

    H = torch.sum((yi * alpha + 1).view(-1,1,1) * xij * k.view(-1,1,1) / gamma_sq.view(-1,1,1), dim = 0)
    return torch.inverse(H)

def compute_hessians(input_matrix, model_matrix, coeff, overdispersion, full_cov=True):
    n_samples, n_genes = input_matrix.shape
    n_coefficients = model_matrix.shape[1]
    loc_shape = (n_genes, n_coefficients, n_coefficients) if full_cov else (n_genes, n_coefficients)
    loc = torch.zeros(loc_shape)

    t = trange(n_genes)
    for gene_idx in t:
        solved_hessian = compute_hessian(obs=input_matrix[:,gene_idx], model_matrix=model_matrix, coeff=coeff[:,gene_idx], overdispersion=overdispersion[gene_idx])

        t.set_description('Variance estimation: {:.2f}  '.format(gene_idx / n_genes))
        t.refresh()

        if full_cov:
            loc[gene_idx, :, :] = solved_hessian
        else:
            loc[gene_idx, :] = torch.diag(solved_hessian)

    return loc

def compute_sandwiches(input_matrix, model_matrix, coeff, overdispersion, size_factors, full_cov = True):
    n_samples, n_genes = input_matrix.shape
    n_coefficients = model_matrix.shape[1]
    loc_shape = (n_genes, n_coefficients, n_coefficients) if full_cov else (n_genes, n_coefficients)
    loc = torch.zeros(loc_shape)

    t = trange(n_genes)
    for gene_idx in t:
        sandwich = compute_bread_and_meat(obs=input_matrix[:,gene_idx], model_matrix=model_matrix, coeff=coeff[:,gene_idx], overdispersion=overdispersion[gene_idx], size_factors=size_factors)

        t.set_description('Variance estimation: {:.2f}  '.format(gene_idx / n_genes))
        t.refresh()

        if full_cov:
            loc[gene_idx, :, :] = sandwich
        else:
            loc[gene_idx, :] = torch.diag(sandwich)

    return loc


def compute_bread_and_meat(obs, model_matrix, coeff, overdispersion, size_factors):
    beta = torch.tensor(coeff)
    model_matrix = torch.tensor(model_matrix)
    alpha = overdispersion
    design_v = model_matrix.t()  # Transpose for vectorized operations
    yi = obs.unsqueeze(1)  # Add a new axis for broadcasting
    k = size_factors * torch.exp(torch.matmul(design_v.t(), beta))
    gamma_sq = (1 + alpha * k) ** 2

    xij = torch.einsum('ik,jk->ijk', design_v, design_v).permute(2,0,1)

    bread = torch.sum((yi * alpha + 1).view(-1,1,1) * xij * k.view(-1,1,1) / gamma_sq.view(-1,1,1), dim = 0)
    bread = torch.inverse(bread)
    
    meat = torch.sum(((xij * yi.view(-1,1,1)) + (yi + 1 / alpha).view(-1,1,1) * (xij * k.view(-1,1,1) * alpha.view(-1,1,1)) / (gamma_sq.view(-1,1,1)))**2, dim = 0)

    return torch.matmul(torch.matmul(bread, meat), bread)

# Example usage:
# input_matrix, model_matrix, coeff, overdispersion = ...
# result = compute_hessians(input_matrix, model_matrix, coeff, overdispersion, full_cov=True)
