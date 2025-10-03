# FedWBA: Personalized Bayesian Federated Learning with Wasserstein Barycenter Aggregation
# Copyright (C) 2025  Ting Wei

import torch
import numpy as np

def compute_log_likelihood(model, x, y):

    prediction = model(x)
    log_like_data = torch.sum(torch.log(prediction)[torch.arange(y.shape[0]), y])
    return log_like_data


def compute_gradients(model, x, y):

    log_likelihood = compute_log_likelihood(model, x, y)
    log_likelihood.backward()
    grads = [param.grad.clone().flatten() for param in model.parameters()]
    for grad in grads:
        if torch.isnan(grad).any():
            return torch.zeros_like(torch.cat(grads))
    grads_tensor = torch.cat(grads)
    model.zero_grad()

    return grads_tensor


def svgd_update(M, particles, grad_logp, kxy, dxkxy, sum_squared_grad, alpha_ada, epsilon_ada, betta):
    
    kxy = torch.tensor(kxy, dtype=torch.float32)
    grad_logp = torch.tensor(grad_logp, dtype=torch.float32)
    delta_theta = (1 / M) * (torch.mm(kxy, grad_logp) + dxkxy)

    if sum_squared_grad is None:
        sum_squared_grad = torch.pow(delta_theta.detach().clone(), 2)
    else:
        sum_squared_grad = betta * sum_squared_grad + (1 - betta) * torch.pow(delta_theta.detach().clone(), 2)
    
    epsilon_svgd = alpha_ada / (epsilon_ada + torch.sqrt(sum_squared_grad))
    with torch.no_grad():
        particles = particles + epsilon_svgd * delta_theta.detach().clone()
    
    return particles, sum_squared_grad