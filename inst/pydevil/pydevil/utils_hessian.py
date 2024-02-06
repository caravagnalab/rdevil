import torch
from tqdm import trange

def compute_hessian(obs, model_matrix, coeff, overdispersion, size_factors):
    beta = torch.tensor(coeff)
    model_matrix = torch.tensor(model_matrix)
    alpha = 1 / overdispersion
    design_v = model_matrix.t()  # Transpose for :qvectorized operations
    yi = obs.unsqueeze(1)  # Add a new axis for broadcasting
    k = size_factors * torch.exp(torch.matmul(design_v.t(), beta))
    gamma_sq = (1 + alpha * k) ** 2

    xij = torch.einsum('ik,jk->ijk', design_v, design_v).permute(2,0,1)

    H = torch.sum((yi * alpha + 1).view(-1,1,1) * xij * k.view(-1,1,1) / gamma_sq.view(-1,1,1), dim = 0)
    return torch.inverse(H)

def compute_hessians(input_matrix, model_matrix, coeff, overdispersion, size_factors, full_cov=True):
    n_samples, n_genes = input_matrix.shape
    n_coefficients = model_matrix.shape[1]
    loc_shape = (n_genes, n_coefficients, n_coefficients) if full_cov else (n_genes, n_coefficients)
    loc = torch.zeros(loc_shape)

    t = trange(n_genes)
    for gene_idx in t:
        solved_hessian = compute_hessian(
            obs=input_matrix[:,gene_idx],
            model_matrix=model_matrix,
            coeff=coeff[:,gene_idx],
            overdispersion=overdispersion[gene_idx],
            size_factors=size_factors
        )

        t.set_description('Variance estimation: {:.2f}  '.format(gene_idx / n_genes))
        t.refresh()
        if torch.cuda.is_available():
            solved_hessian = solved_hessian.detach().cpu()

        if full_cov:
            loc[gene_idx, :, :] = solved_hessian
        else:
            loc[gene_idx, :] = torch.diag(solved_hessian)
        del solved_hessian

    return loc

def compute_bread(obs, model_matrix, coeff, overdispersion, size_factors):
    beta = torch.tensor(coeff)
    model_matrix = torch.tensor(model_matrix)
    alpha = 1 / overdispersion
    design_v = model_matrix.t()  # Transpose for vectorized operations
    yi = obs.unsqueeze(1)  # Add a new axis for broadcasting
    k = size_factors * torch.exp(torch.matmul(design_v.t(), beta))
    gamma_sq = (1 + alpha * k) ** 2

    xij = torch.einsum('ik,jk->ijk', design_v, design_v).permute(2,0,1)

    H = torch.sum((yi * alpha + 1).view(-1,1,1) * xij * k.view(-1,1,1) / gamma_sq.view(-1,1,1), dim = 0)
    return torch.inverse(H) * obs.shape[0]

def compute_scores(design_matrix, y, beta, alpha):
    xmat = design_matrix
    mu = torch.exp(xmat @ beta)
    r = (y - mu) / mu

    v = mu + mu.pow(2) / alpha
    w = mu.pow(2) / v

    return xmat * r.view(-1, 1) * w.view(-1, 1)

def compute_sandwich(design_matrix, y, beta, alpha, size_factors, cluster):
    b = compute_bread(y, design_matrix, beta, 1 / alpha, size_factors)
    m = compute_clustered_meat(design_matrix, y, beta, 1 / alpha, cluster)
    return (b @ m @ b) / y.shape[0]


def compute_sandwiches(input_matrix, model_matrix, coeff, overdispersion, size_factors, cluster):
    n_samples, n_genes = input_matrix.shape
    n_coefficients = model_matrix.shape[1]
    loc_shape = (n_genes, n_coefficients, n_coefficients)
    loc = torch.zeros(loc_shape)

    t = trange(n_genes)
    for gene_idx in t:
        s = compute_sandwich(
            design_matrix=model_matrix, 
            y=input_matrix[:,gene_idx], 
            beta=coeff[:,gene_idx], 
            alpha=overdispersion[gene_idx], 
            size_factors=size_factors, 
            cluster=cluster
        )

        t.set_description('Clustered variance estimation: {:.2f}  '.format(gene_idx / n_genes))
        t.refresh()

        if torch.cuda.is_available():
            s = s.detach().cpu()

        loc[gene_idx, :, :] = s
        del s

    return loc

def compute_clustered_meat(design_matrix, y, beta, alpha, cluster=None):
    ef = compute_scores(design_matrix, y, beta, alpha)
    k = ef.shape[1]
    n = ef.shape[0]

    rval = torch.zeros((k, k))  # Assuming you want to keep the dimnames

    if cluster is None:
        cluster = torch.arange(1, n + 1)

    if cluster is None:
        raise ValueError("Cluster not specified.")

    cluster = torch.as_tensor(cluster)

    if len(cluster) != n:
        raise ValueError("Number of observations in 'cluster' and in 'y' do not match")

    if cluster.isnan().any():
        raise ValueError("Cannot handle NAs in 'cluster'")

    cl = [1]
    sign = 1

    g = [len(torch.unique(cluster))]

    for i in range(len(cl)):
        efi = ef.clone()  # Make a copy to avoid modifying the original

        adj = g[i] / (g[i] - 1) if g[i] > 1 else 1

        if g[i] < n:
            efi = torch.zeros(g[i], k)

            for i, group in enumerate(torch.unique(cluster)):
                mask = cluster == group
                efi[i,:] += ef[mask].sum(dim=0)
            
        rval += sign * adj * torch.mm(efi.t(), efi) / n

    return rval



