# FedWBA: Personalized Bayesian Federated Learning with Wasserstein Barycenter Aggregation
# Copyright (C) 2025  Ting Wei

import numpy as np
import torch

def pairwise_distances(N, x, y):

    """
    Returns distance matrix that contains the distance between each particle (column) in x and y
    """
    
    x = x.float()  
    y = y.float() 
    x_squared = x.norm(dim=0)**2
    new_x = x_squared.view(1, -1).repeat(N, 1).transpose(0, 1)
    y_squared = y.norm(dim=0)**2
    new_y = y_squared.view(1, -1).repeat(N, 1)
    return new_x + new_y - 2 * torch.mm(x.transpose(0, 1), y)


def kde(N, d, my_lambda, distances_squared, kernel='gaussian'):

    """
    KDE over d dimensional particles
    """

    if kernel == 'gaussian':
        exp_distances = torch.exp(distances_squared*(-0.5)*(1/my_lambda**2)) + 10**(-9)
        sum_exp_distances = exp_distances.sum(dim=1)
        return torch.exp(torch.log(sum_exp_distances + 10**(-50)) - (d/2)*np.log(N*(np.pi*2*my_lambda**2) + 10**(-50)))



def svgd_kernel(theta, kernel='gaussian', h=-1):

    """
    This function is borrowed from SVGD original paper code accessible at
    https://github.com/DartML/Stein-Variational-Gradient-Descent.
    Returns RBF kernel matrix and its derivative
    """

    pairwise_dists = torch.cdist(theta, theta)
    pairwise_dists_squared = pairwise_dists ** 2

    if h < 0:
        h = torch.median(pairwise_dists_squared)
        h = torch.sqrt(0.5 * h / torch.log(torch.tensor(theta.shape[0] + 1)))

    if kernel == 'gaussian':
        Kxy = torch.exp(-pairwise_dists_squared / (h ** 2) / 2)

        sumkxy = torch.sum(Kxy, dim=1).view(-1, 1)
        dxkxy = -torch.matmul(Kxy, theta) + theta * sumkxy
        dxkxy = dxkxy / (h ** 2)

    elif kernel == 'laplacian':
        Kxy = torch.exp(-pairwise_dists / h)

        dxkxy = torch.zeros_like(theta)
        for i in range(theta.shape[0]):
            diff = theta[i] - theta
            dist = pairwise_dists[i].unsqueeze(-1)
            dist[dist == 0] = 1e-10  
            dxkxy[i] = torch.sum((Kxy[i].unsqueeze(-1) * diff / dist / h), dim=0)

    elif kernel == 'sigmoid':
        alpha = 1 
        c = 0.0  
        inner_product_matrix = torch.matmul(theta, theta.T)
        Kxy = torch.tanh(alpha * inner_product_matrix + c)

        dxkxy = torch.zeros_like(theta)
        for i in range(theta.shape[0]):
            sech_squared = 1 - torch.tanh(alpha * inner_product_matrix[i] + c) ** 2
            dxkxy[i] = torch.sum(alpha * sech_squared.unsqueeze(-1) * theta, dim=0)

    elif kernel == 'polynomial':
        c = 0  
        d = 2  
        inner_product_matrix = torch.matmul(theta, theta.T)
        Kxy = (inner_product_matrix + c) ** d

        dxkxy = torch.zeros_like(theta)
        for i in range(theta.shape[0]):
            power_term = d * (inner_product_matrix[i] + c) ** (d - 1)
            dxkxy[i] = torch.sum(power_term.unsqueeze(-1) * theta, dim=0)

    return (Kxy, dxkxy)


def compute_accuracy(net, X_test, y_test, particles):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracies = []
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    for particle in particles:
        net.set_net_param(particle.to(device))
        with torch.no_grad():
            outputs = net(X_test)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_test).sum().item()
            accuracy = correct / y_test.size(0) * 100
            accuracies.append(accuracy)

    max_accuracy = max(accuracies)
    mean_accuracy = sum(accuracies) / len(accuracies)
    min_accuracy = min(accuracies)
    return max_accuracy, mean_accuracy, min_accuracy


def forward_fill(data):
    
    filled_data = []
    last_non_zero = None
    for value in data:
        if value != 0:
            last_non_zero = value
        filled_data.append(last_non_zero if last_non_zero is not None else value)
    return filled_data


def client_average_acc(num_global,local_acc):

    data_filled = {key: forward_fill(value) for key, value in local_acc.items()}
    num_lists = len(data_filled)
    average_list = [0.0] * num_global
    
   
    for lst in data_filled.values():
        for i in range(num_global):
            average_list[i] += lst[i]
    
    
    for i in range(num_global):
        average_list[i] /= num_lists
    
    return average_list