# FedWBA: Personalized Bayesian Federated Learning with Wasserstein Barycenter Aggregation
# Copyright (C) 2025  Ting Wei

from ot.lp import emd
import torch


def dist(x1, x2):
    
    """
    Compute the squared Euclidean distance between samples in x1 and x2.
 
    Args:
    ----------
    x1 : torch.Tensor,  (n1, d)
      
    x2 : torch.Tensor,  (n2, d)
       
 
    Returns
    -------
    M : torch.Tensor, shape (n1, n2)
        Distance matrix computed with squared Euclidean distance.
    """
    
    x1 = torch.tensor(x1, dtype=torch.float)
    x2 = torch.tensor(x2, dtype=torch.float)
    
    a2 = torch.sum(x1 ** 2, dim=1)  
    b2 = torch.sum(x2 ** 2, dim=1)  
    c = -2 * torch.mm(x1, x2.t())   
    
    distances = a2[:, None] + b2[None, :] + c
    
    return distances

def barycenter(measures_locations, measures_weights, X_init, b=None, weights=None, \
               numItermax=100, stopThr=1e-7, verbose=False, record=None):
    
     
    '''
    Args:
    ----
    measures_locations: A list containing N arrays of shape (k_i, d), where each array represents a d-dimensional support, and k_i is the number of support points.
    measures_weights: A list containing N arrays of shape (k_i,), where each array represents the internal weights of an input measure, i.e., the weight on each support point.
    X_init: The initial locations of the support, with shape (k, d), where k is the number of support points and can be freely set.
    
    b: The initial weights of the barycenter. By default, each support point on the barycenter has the same weight, i.e., 1 divided by the number of samples.
    weight: The weights for solving the barycenter of different measures. By default, each input measure has the same weight, i.e., 1 divided by the number of measures.
    
    Returns:
    --------
    torch.Tensor: The optimized barycenter as the global particle.
    
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    iter_count = 0
    
    N = len(measures_locations) 
    k = X_init.shape[0] 
    d = X_init.shape[1]
    
    if b is None:
        b = torch.ones((k,), dtype=X_init.dtype) / k
    if weights is None:
        weights = torch.ones((N,), dtype=X_init.dtype) / N
    b = b.to(device)
    weights = weights.to(device)

    X = X_init.clone().to(device)
    
    record_dict = {}
    displacement_square_norms = []

    displacement_square_norm = stopThr + 1
    
    while (displacement_square_norm > stopThr and iter_count < numItermax):
        
        update = torch.zeros((k, d), dtype=X_init.dtype).to(device)
        
        for (measure_locations_i, measure_weights_i, weight_i) in zip(measures_locations, measures_weights, weights):
            
            measure_locations_i = measure_locations_i.to(device)
            measure_weights_i = measure_weights_i.to(device)
            weight_i = weight_i.to(device)
            
            M_i = dist(X, measure_locations_i)
            
            T_i = emd(b, measure_weights_i, M_i) 

            update += weight_i * 1. / b[:, None] * torch.matmul(T_i, measure_locations_i)
            
        displacement_square_norm = torch.sum((update - X) ** 2)
        if record:
            displacement_square_norms.append(displacement_square_norm)

        X = update
        
        # if verbose:
            # print('iteration:{}, displacement_square_norm={}'.format(iter_count, displacement_square_norm))

        iter_count += 1
        
    if record:
        record_dict['displacement_square_norms'] = displacement_square_norms
            
        return X, record_dict

    else:
        return X