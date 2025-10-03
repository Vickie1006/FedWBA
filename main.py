#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import time
import os
import argparse

from datasets.data import data_split
from model.network import MLP as MLP
from model.network import LeNet as LeNet
from optim.general_functions import client_average_acc
from server.server_parallel import server


def write_parameters(param_dir, dataset, num_exp, num_svgd, num_devices, M, k, batch_size, alpha_ada, epsilon_ada, betta, h):
    
    
    dir_path = os.path.dirname(param_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    param_info = (
        f"Dataset: {dataset}\n"
        f"Number of experiments: {num_exp}\n"
        f"Number of svgd iterations on the client side: {num_svgd}\n"
        f"Number of devices: {num_devices}\n"
        f"Number of particles: {M}\n"
        f"Number of clients scheduled per round: {k}\n"
        f"Batch size: {batch_size}\n"
        f"Alpha_ada: {alpha_ada}\n"
        f"Epsilon_ada: {epsilon_ada}\n"
        f"Betta: {betta}\n"
        f"Kernel bandwidth: {h}\n"
    )
    with open(param_dir, 'a') as f:
        f.write("\n\n--- Parameters at the beginning of the experiment ---\n")
        f.write(param_info)


def main(dataset, num_exp, num_svgd, num_devices, M, k, batch_size, alpha_ada, epsilon_ada, betta, h):

    result_dir = 'result/{}/{} clients'.format(dataset, num_devices)
    param_dir = os.path.join(result_dir, 'FedWBA.txt')

    # seed
    torch.random.manual_seed(42)
    np.random.seed(42)

    # fixed parameter
    num_global = 101    
        
    write_parameters(param_dir, dataset, num_exp, num_svgd, num_devices, M, k, batch_size, alpha_ada, epsilon_ada, betta, h)
    
    for exp in range(num_exp):
        print('Trial {}'.format(exp+1))
        start_time = time.time()

        client_data = data_split(
            num_devices, 
            num_labels = 5, 
            dataset_type = dataset, 
            check = False
        )

        if dataset == "MNIST" or dataset == "FashionMNIST":
            model = MLP(num_hidden=100)
        elif dataset == "CIFAR10":
            model = LeNet()
        elif dataset == "CIFAR100":
            model = LeNet()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")


        local_acc, local_acc_max = server(M, model, k, num_svgd, num_global, client_data, batch_size, alpha_ada, epsilon_ada, betta, h)


        end_time = time.time()
        elapsed_time = end_time - start_time
        print('-----------------------------------')
        print(f"Time required for trial {exp+1}: {elapsed_time} s")
        print('==============Summary==============')
        client_acc = client_average_acc(num_global,local_acc)
        print('The average accuracy on client is:', max(client_acc))
        print(' ')

        local_acc_filename = os.path.join(result_dir, 'FedWBA.txt')
        dir_path = os.path.dirname(local_acc_filename)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            
        with open(local_acc_filename, 'a') as f:
            f.write(f'Trial{exp+1} {client_acc}\n')

    print('Finish')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='Dataset name')
    parser.add_argument('--num_exp', type=int, default=1, help='Number of experiments')
    parser.add_argument('--num_svgd', type=int, default=30, help='Number of svgd iterations on the client side')
    parser.add_argument('--num_devices', type=int, default=100, help='Number of devices')
    parser.add_argument('--M', type=int, default=10, help='Number of particles')
    parser.add_argument('--k', type=int, default=10, help='Number of clients scheduled per round')
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--alpha_ada', type=float, default=10 ** (-3), help='Alpha_ada parameter')
    parser.add_argument('--epsilon_ada', type=float, default=10**(-8), help='Epsilon_ada parameter')
    parser.add_argument('--betta', type=float, default=0.9, help='Betta parameter')
    parser.add_argument('--bandwidth', type=float, default=-1, help='kernel bandwidth in SVGD')


    args = parser.parse_args()

    print("Received parameters:")
    print(f"Dataset: {args.dataset}")
    print(f"Number of experiments: {args.num_exp}")
    print(f"Number of svgd iterations on the client side: {args.num_svgd}")
    print(f"Number of devices: {args.num_devices}")
    print(f"Number of particles: {args.M}")
    print(f"Number of clients scheduled per round: {args.k}")
    print(f"Batch size: {args.batch_size}")
    print(f"Alpha_ada: {args.alpha_ada}")
    print(f"Epsilon_ada: {args.epsilon_ada}")
    print(f"Betta: {args.betta}")
    print(f"Bandwidth: {args.bandwidth}")

    
    if args.dataset == "MNIST" or args.dataset == "FashionMNIST":
        model = MLP(num_hidden=100)
    elif args.dataset == "CIFAR10":
        model = LeNet()
    elif args.dataset == "CIFAR100":
        model = LeNet(num_classes=100)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


    print("Model architecture:")
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")


    main(args.dataset, args.num_exp, args.num_svgd, args.num_devices, args.M, args.k, args.batch_size, args.alpha_ada, args.epsilon_ada, args.betta, args.bandwidth)