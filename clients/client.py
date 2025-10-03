# FedWBA: Personalized Bayesian Federated Learning with Wasserstein Barycenter Aggregation
# Copyright (C) 2025  Ting Wei

import torch
from optim.SVGD import compute_gradients, svgd_update
from optim.general_functions import pairwise_distances, kde, svgd_kernel

my_lambda=0.55

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
def client(i, net, num_svgd, y_train, X_train, global_particles, local_particles, batch_size, alpha_ada, epsilon_ada, betta, h):
    
    """
    Client-side update function for FedWBA algorithm.
    
    Performs local SVGD (Stein Variational Gradient Descent) updates on client particles,
    incorporating global particles through Wasserstein barycenter aggregation.
    
    Args:
        i: Client index
        net: Neural network model
        num_svgd: Number of SVGD iterations to perform
        global_particles: Particles aggregated from the server/global model
        local_particles: Initial local particles for the client
        batch_size: Batch size for stochastic gradient updates
        alpha_ada: Learning rate parameter for adaptive updates
        epsilon_ada: Numerical stability parameter for adaptive updates
        betta: Momentum parameter for SVGD updates
        h: Bandwidth parameter for kernel
        
    Returns:
        Updated local particles after SVGD iterations
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device) 
    
    # Get model parameters
    num_vars = net.num_vars
    M = len(local_particles)  # Number of particles
    sum_squared_grad = torch.zeros([M, net.num_vars]).to(device)
    
    if not isinstance(y_train, torch.Tensor) or (isinstance(y_train, torch.Tensor) and y_train.dtype!= torch.int64):
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    else:
        y_train = y_train.to(device)
    
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.to(device)

    if i == 0:
        particles_2 = local_particles.detach().clone().to(device)
       
    else:
        
        particles_2 = local_particles.detach().clone().requires_grad_(True).to(device)
        
        for _ in range(0, num_svgd):
        
            grad_theta = torch.zeros([M, net.num_vars]).to(device)
            
            # Compute kernel and its gradients
            kxy, dxkxy = svgd_kernel(particles_2.detach().clone(), h=h)
            
            distance_M_i_1 = pairwise_distances(M, particles_2.T, global_particles.T)
            distance_M_i_1 = distance_M_i_1.to(device)
            qi_1 = kde(M, num_vars, my_lambda, distance_M_i_1, 'gaussian')
            sv_target = torch.log(qi_1+ 10**(-10)).to(device)
            
            indices = torch.randperm(len(X_train))[:batch_size]
            X_train_batch = X_train[indices]
            y_train_batch = y_train[indices]
            
            for m in range(M):
                net.set_net_param(particles_2[m])
                grad_theta[m,:] = compute_gradients(net, X_train_batch, y_train_batch)
                
            sv_target.backward(torch.ones(M).to(device))
            grad_sv_target = particles_2.grad
            grad_logp =  grad_theta + grad_sv_target 
            
            particles_2, sum_squared_grad = svgd_update(M, particles_2, grad_logp, kxy, dxkxy, sum_squared_grad, alpha_ada, epsilon_ada, betta)
            particles_2.requires_grad = True

        particles_2.requires_grad = False
            
    return particles_2