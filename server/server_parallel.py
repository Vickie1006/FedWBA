# FedWBA: Personalized Bayesian Federated Learning with Wasserstein Barycenter Aggregation
# Copyright (C) 2025  Ting Wei

import numpy as np
import torch
import torch.multiprocessing as mp
from clients.client import client
from optim.general_functions import compute_accuracy
from server.bary import barycenter

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def client_wrapper(args):
    i, client_idx, net, num_svgd, batch_size, client_data, global_particles, local_particles, alpha_ada, epsilon_ada, betta, h = args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_train_X, client_train_y = client_data[client_idx][0].to(device), client_data[client_idx][1].to(device)
    client_test_X, client_test_y = client_data[client_idx][2].to(device), client_data[client_idx][3].to(device)

    updated_local_particles = client(
        i,
        net,
        num_svgd,
        client_train_y,
        client_train_X,
        global_particles.float(),
        local_particles[client_idx],
        batch_size,
        alpha_ada, 
        epsilon_ada, 
        betta,
        h
    )
    local_particles[client_idx] = updated_local_particles

  
    train_max_acc, train_mean_acc, train_min_acc = compute_accuracy(
        net, client_train_X, client_train_y, local_particles[client_idx]
    )
    print(
    "Client {}: Accuracy of local particles on the training set:\nmax: {:.2f}, mean: {:.2f}, min: {:.2f}".format(
        client_idx+1, train_max_acc, train_mean_acc, train_min_acc
        )
    )
   
    local_acc_max, local_acc, test_min_acc = compute_accuracy(
        net, client_test_X, client_test_y, local_particles[client_idx]
    )
    print(
    "Client {}: Accuracy of local particles on the test set:\nmax: {:.2f}, mean: {:.2f}".format(
        client_idx+1, local_acc_max, local_acc
        )
    )
    print('')

    return (
        client_idx,
        local_particles[client_idx].clone().cpu(),
        local_acc_max,  
        local_acc,  
    )

def server(M, net, k, num_svgd, num_global, client_data, batch_size, alpha_ada, epsilon_ada, betta, h):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)  
   
    num_devices = len(client_data)
    num_vars = net.num_vars

    local_particles = []
    for i in range(num_devices):
        particles = net.initialize(M) 
        local_particles.append(particles)
    local_particles = torch.stack(local_particles).to(device)
    global_particles = torch.mean(local_particles, dim=0).detach().clone().to(device)

    local_acc = {i: [0] * num_global for i in range(1, num_devices + 1)}
    local_acc_max = {i: [0] * num_global for i in range(1, num_devices + 1)}

    for i in range(0, num_global):
        print("================Rounds of communication: {}================".format(i))

        start_client = i * k % num_devices + 1
        clients_this_round = [(start_client + j - 1) % num_devices + 1 for j in range(k)]
        ids = [x - 1 for x in clients_this_round]  
        
        num_processes = min(len(ids), 5)
        with mp.get_context("spawn").Pool(processes=num_processes) as pool:
            args = [
                (
                    i,
                    client_idx,
                    net,
                    num_svgd,
                    batch_size,
                    client_data,
                    global_particles,
                    local_particles,
                    alpha_ada, 
                    epsilon_ada, 
                    betta,
                    h
                )
                for client_idx in ids
            ]
            results = pool.map(client_wrapper, args)

        
        for client_idx, updated_local_particles, local_acc_max_val, local_acc_val in results:
            local_particles[client_idx] = updated_local_particles.to(device)
            local_acc_max[client_idx + 1][i] = local_acc_max_val  
            local_acc[client_idx + 1][i] = local_acc_val  

        num_samples = len(particles)
        weights = [
            torch.ones(num_samples, dtype=torch.float32).to(device) / num_samples
            for _ in range(num_devices)
        ]
        x_init = local_particles.mean(dim=0)  
        global_particles = barycenter(
            [local_particles[i, :, :] for i in range(num_devices)],
            weights,  # measures_weights
            x_init,
            verbose=True,
        ).to(device)

        
    

    return local_acc, local_acc_max
