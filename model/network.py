# FedWBA: Personalized Bayesian Federated Learning with Wasserstein Barycenter Aggregation
# Copyright (C) 2025  Ting Wei

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class MLP(nn.Module):
    def __init__(self, num_hidden, num_classes=10):
        super().__init__()
        input_dim = 28 * 28
        self.fc1 = nn.Linear(input_dim, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_classes)
        self.num_vars = self._calculate_num_vars()  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _calculate_num_vars(self):
        return self.fc1.weight.numel() + self.fc1.bias.numel() + self.fc2.weight.numel() + self.fc2.bias.numel()

    def initialize(self, M):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        all_flattened_params = []
        for _ in range(M):
            fc1_weight = nn.init.kaiming_uniform_(torch.empty(self.fc1.weight.shape), nonlinearity='relu').to(device)
            fc1_bias = torch.zeros_like(self.fc1.bias).to(device)
            fc2_weight = nn.init.kaiming_uniform_(torch.empty(self.fc2.weight.shape), nonlinearity='relu').to(device)
            fc2_bias = torch.zeros_like(self.fc2.bias).to(device)

            flattened_params = torch.cat([
                fc1_weight.flatten(),
                fc1_bias.flatten(),
                fc2_weight.flatten(),
                fc2_bias.flatten()
            ]).unsqueeze(0)
            all_flattened_params.append(flattened_params)

        flattened_params_tensor = torch.cat(all_flattened_params, dim=0)

        for param in self.parameters():
            param.data.zero_()

        return flattened_params_tensor

    def forward(self, x):
        if isinstance(x, torch.Tensor) and x.dtype == torch.uint8:
            
            x = x.float().to(self.device)
        elif isinstance(x, torch.Tensor) and x.device.type!= self.device.type:
            x = x.to(self.device)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

    def set_net_param(self, particles):
        old_fc1_weight = self.fc1.weight.data.clone()
        old_fc1_bias = self.fc1.bias.data.clone()
        old_fc2_weight = self.fc2.weight.data.clone()
        old_fc2_bias = self.fc2.bias.data.clone()

        fc1_weight_size = self.fc1.weight.numel()
        fc1_bias_size = self.fc1.bias.numel()
        fc2_weight_size = self.fc2.weight.numel()
        fc2_bias_size = self.fc2.bias.numel()

        fc1_weight = particles[:fc1_weight_size].reshape(self.fc1.weight.shape)
        fc1_bias = particles[fc1_weight_size:fc1_weight_size + fc1_bias_size].reshape(self.fc1.bias.shape)
        fc2_weight = particles[fc1_weight_size + fc1_bias_size:fc1_weight_size + fc1_bias_size + fc2_weight_size].reshape(
            self.fc2.weight.shape)
        fc2_bias = particles[fc1_weight_size + fc1_bias_size + fc2_weight_size:].reshape(self.fc2.bias.shape)

        self.fc1.weight.data = torch.tensor(fc1_weight, dtype=torch.float32, requires_grad=True)
        self.fc1.bias.data = torch.tensor(fc1_bias, dtype=torch.float32, requires_grad=True)
        self.fc2.weight.data = torch.tensor(fc2_weight, dtype=torch.float32, requires_grad=True)
        self.fc2.bias.data = torch.tensor(fc2_bias, dtype=torch.float32, requires_grad=True)

        if (self.fc1.weight.data.equal(old_fc1_weight) and
                self.fc1.bias.data.equal(old_fc1_bias) and
                self.fc2.weight.data.equal(old_fc2_weight) and
                self.fc2.bias.data.equal(old_fc2_bias)):
            print("Note that the network parameters have not been modified!")
        else:
            pass


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
        self.num_vars = self._calculate_num_vars()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _calculate_num_vars(self):
        base_conv1_weight_size = self.base[0].weight.numel()
        base_conv1_bias_size = self.base[0].bias.numel()
        base_conv2_weight_size = self.base[3].weight.numel()  
        base_conv2_bias_size = self.base[3].bias.numel()  
        head_fc1_weight_size = self.head[0].weight.numel()
        head_fc1_bias_size = self.head[0].bias.numel()
        head_fc2_weight_size = self.head[2].weight.numel()
        head_fc2_bias_size = self.head[2].bias.numel()
        head_fc3_weight_size = self.head[4].weight.numel()
        head_fc3_bias_size = self.head[4].bias.numel()
        return (base_conv1_weight_size + base_conv1_bias_size + base_conv2_weight_size + base_conv2_bias_size +
                head_fc1_weight_size + head_fc1_bias_size + head_fc2_weight_size + head_fc2_bias_size +
                head_fc3_weight_size + head_fc3_bias_size)

    def initialize(self, M):
        all_flattened_params = []
        for _ in range(M):
            base_conv1_weight = nn.init.xavier_uniform_(torch.empty(self.base[0].weight.shape)).to(self.device)
            base_conv1_bias = torch.zeros_like(self.base[0].bias).to(self.device)
            base_conv2_weight = nn.init.xavier_uniform_(torch.empty(self.base[3].weight.shape)).to(self.device)  # 修改：原来为 self.base[2]，现在为 self.base[3]
            base_conv2_bias = torch.zeros_like(self.base[3].bias).to(self.device)  # 修改：原来为 self.base[2]，现在为 self.base[3]
            head_fc1_weight = nn.init.xavier_uniform_(torch.empty(self.head[0].weight.shape)).to(self.device)
            head_fc1_bias = torch.zeros_like(self.head[0].bias).to(self.device)
            head_fc2_weight = nn.init.xavier_uniform_(torch.empty(self.head[2].weight.shape)).to(self.device)
            head_fc2_bias = torch.zeros_like(self.head[2].bias).to(self.device)
            head_fc3_weight = nn.init.xavier_uniform_(torch.empty(self.head[4].weight.shape)).to(self.device)
            head_fc3_bias = torch.zeros_like(self.head[4].bias).to(self.device)

            flattened_params = torch.cat([
                base_conv1_weight.flatten(),
                base_conv1_bias.flatten(),
                base_conv2_weight.flatten(),
                base_conv2_bias.flatten(),
                head_fc1_weight.flatten(),
                head_fc1_bias.flatten(),
                head_fc2_weight.flatten(),
                head_fc2_bias.flatten(),
                head_fc3_weight.flatten(),
                head_fc3_bias.flatten()
            ]).unsqueeze(0)
            all_flattened_params.append(flattened_params)

        flattened_params_tensor = torch.cat(all_flattened_params, dim=0)

        for param in self.parameters():
            param.data.zero_()

        return flattened_params_tensor

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = np.transpose(x, (0, 3, 1, 2))
            x = torch.from_numpy(x).float().to(self.device)
        elif isinstance(x, torch.Tensor) and x.device.type!= self.device.type:
            x = x.to(self.device)

        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return F.softmax(x, dim=1)

    def set_net_param(self, particles):
        old_base_conv1_weight = self.base[0].weight.data.clone()
        old_base_conv1_bias = self.base[0].bias.data.clone()
        old_base_conv2_weight = self.base[3].weight.data.clone() 
        old_base_conv2_bias = self.base[3].bias.data.clone()  
        old_head_fc1_weight = self.head[0].weight.data.clone()
        old_head_fc1_bias = self.head[0].bias.data.clone()
        old_head_fc2_weight = self.head[2].weight.data.clone()
        old_head_fc2_bias = self.head[2].bias.data.clone()
        old_head_fc3_weight = self.head[4].weight.data.clone()
        old_head_fc3_bias = self.head[4].bias.data.clone()

        base_conv1_weight_size = self.base[0].weight.numel()
        base_conv1_bias_size = self.base[0].bias.numel()
        base_conv2_weight_size = self.base[3].weight.numel()  
        base_conv2_bias_size = self.base[3].bias.numel() 
        head_fc1_weight_size = self.head[0].weight.numel()
        head_fc1_bias_size = self.head[0].bias.numel()
        head_fc2_weight_size = self.head[2].weight.numel()
        head_fc2_bias_size = self.head[2].bias.numel()
        head_fc3_weight_size = self.head[4].weight.numel()
        head_fc3_bias_size = self.head[4].bias.numel()

        base_conv1_weight = particles[:base_conv1_weight_size].reshape(self.base[0].weight.shape)
        base_conv1_bias = particles[base_conv1_weight_size:base_conv1_weight_size + base_conv1_bias_size].reshape(self.base[0].bias.shape)
        base_conv2_weight = particles[base_conv1_weight_size + base_conv1_bias_size:base_conv1_weight_size + base_conv2_weight_size + base_conv1_bias_size].reshape(
            self.base[3].weight.shape)  
        base_conv2_bias = particles[base_conv1_weight_size + base_conv2_weight_size + base_conv1_bias_size:base_conv1_weight_size + base_conv2_weight_size + base_conv1_bias_size + base_conv2_bias_size].reshape(
            self.base[3].bias.shape)  
        head_fc1_weight = particles[base_conv1_weight_size + base_conv2_weight_size + base_conv1_bias_size + base_conv2_bias_size:base_conv1_weight_size + base_conv2_weight_size + base_conv1_bias_size + base_conv2_bias_size + head_fc1_weight_size].reshape(
            self.head[0].weight.shape)
        head_fc1_bias = particles[base_conv1_weight_size + base_conv2_weight_size + base_conv1_bias_size + base_conv2_bias_size + head_fc1_weight_size:base_conv1_weight_size + base_conv2_weight_size + base_conv1_bias_size + base_conv2_bias_size + head_fc1_weight_size + head_fc1_bias_size].reshape(
            self.head[0].bias.shape)
        head_fc2_weight = particles[base_conv1_weight_size + base_conv2_weight_size + base_conv1_bias_size + base_conv2_bias_size + head_fc1_weight_size + head_fc1_bias_size:base_conv1_weight_size + base_conv2_weight_size + base_conv1_bias_size + base_conv2_bias_size + head_fc1_weight_size + head_fc1_bias_size + head_fc2_weight_size].reshape(
            self.head[2].weight.shape)
        head_fc2_bias = particles[base_conv1_weight_size + base_conv2_weight_size + base_conv1_bias_size + base_conv2_bias_size + head_fc1_weight_size + head_fc1_bias_size + head_fc2_weight_size:base_conv1_weight_size + base_conv2_weight_size + base_conv1_bias_size + base_conv2_bias_size + head_fc1_weight_size + head_fc1_bias_size + head_fc2_weight_size + head_fc2_bias_size].reshape(
            self.head[2].bias.shape)
        head_fc3_weight = particles[base_conv1_weight_size + base_conv2_weight_size + base_conv1_bias_size + base_conv2_bias_size + head_fc1_weight_size + head_fc1_bias_size + head_fc2_weight_size + head_fc2_bias_size:base_conv1_weight_size + base_conv2_weight_size + base_conv1_bias_size + base_conv2_bias_size + head_fc1_weight_size + head_fc1_bias_size + head_fc2_weight_size + head_fc2_bias_size + head_fc3_weight_size].reshape(
            self.head[4].weight.shape)
        head_fc3_bias = particles[base_conv1_weight_size + base_conv2_weight_size + base_conv1_bias_size + base_conv2_bias_size + head_fc1_weight_size + head_fc1_bias_size + head_fc2_weight_size + head_fc2_bias_size + head_fc3_weight_size:base_conv1_weight_size + base_conv2_weight_size + base_conv1_bias_size + base_conv2_bias_size + head_fc1_weight_size + head_fc1_bias_size + head_fc2_weight_size + head_fc2_bias_size + head_fc3_weight_size + head_fc3_bias_size].reshape(
            self.head[4].bias.shape)

        self.base[0].weight.data = torch.tensor(base_conv1_weight, dtype=torch.float32, requires_grad=True).to(self.device)
        self.base[0].bias.data = torch.tensor(base_conv1_bias, dtype=torch.float32, requires_grad=True).to(self.device)
        self.base[3].weight.data = torch.tensor(base_conv2_weight, dtype=torch.float32, requires_grad=True).to(self.device)  
        self.base[3].bias.data = torch.tensor(base_conv2_bias, dtype=torch.float32, requires_grad=True).to(self.device) 
        self.head[0].weight.data = torch.tensor(head_fc1_weight, dtype=torch.float32, requires_grad=True).to(self.device)
        self.head[0].bias.data = torch.tensor(head_fc1_bias, dtype=torch.float32, requires_grad=True).to(self.device)
        self.head[2].weight.data = torch.tensor(head_fc2_weight, dtype=torch.float32, requires_grad=True).to(self.device)
        self.head[2].bias.data = torch.tensor(head_fc2_bias, dtype=torch.float32, requires_grad=True).to(self.device)
        self.head[4].weight.data = torch.tensor(head_fc3_weight, dtype=torch.float32, requires_grad=True).to(self.device)
        self.head[4].bias.data = torch.tensor(head_fc3_bias, dtype=torch.float32, requires_grad=True).to(self.device)

        if (self.base[0].weight.data.equal(old_base_conv1_weight) and
                self.base[0].bias.data.equal(old_base_conv1_bias) and
                self.base[3].weight.data.equal(old_base_conv2_weight) and  
                self.base[3].bias.data.equal(old_base_conv2_bias) and 
                self.head[0].weight.data.equal(old_head_fc1_weight) and
                self.head[0].bias.data.equal(old_head_fc1_bias) and
                self.head[2].weight.data.equal(old_head_fc2_weight) and
                self.head[2].bias.data.equal(old_head_fc2_bias) and
                self.head[4].weight.data.equal(old_head_fc3_weight) and
                self.head[4].bias.data.equal(old_head_fc3_bias)):
            print("Note that the network parameters have not been modified!")
        else:
            pass


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.head = nn.Sequential(
            nn.Conv2d(256, 100, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(100, 100)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.num_vars = self._calculate_num_vars()

        for layer in self.modules():
            if hasattr(layer, 'bias'):
                layer.bias.data = layer.bias.data.float()

    def _calculate_num_vars(self):
        base_layers_param_sizes = []
        head_layers_param_sizes = []
        for layer in self.base:
            if hasattr(layer, 'weight'):
                base_layers_param_sizes.append(layer.weight.numel())
            if hasattr(layer, 'bias'):
                base_layers_param_sizes.append(layer.bias.numel())
        for layer in self.head:
            if hasattr(layer, 'weight'):
                head_layers_param_sizes.append(layer.weight.numel())
            if hasattr(layer, 'bias'):
                head_layers_param_sizes.append(layer.bias.numel())
        return sum(base_layers_param_sizes + head_layers_param_sizes)

    def initialize(self, M):
        all_flattened_params = []
        for _ in range(M):
            base_layers_params = []
            head_layers_params = []
            for layer in self.base:
                if hasattr(layer, 'weight'):
                    weight = nn.init.kaiming_uniform_(torch.empty(layer.weight.shape, device=self.device, dtype=torch.float), nonlinearity='relu')
                    base_layers_params.append(weight.flatten())
                    layer.weight.data.zero_()
                if hasattr(layer, 'bias'):
                    bias = torch.zeros_like(layer.bias, device=self.device, dtype=torch.float)
                    base_layers_params.append(bias.flatten())
                    layer.bias.data.copy_(bias)
            for layer in self.head:
                if hasattr(layer, 'weight'):
                    weight = nn.init.kaiming_uniform_(torch.empty(layer.weight.shape, device=self.device, dtype=torch.float), nonlinearity='relu')
                    head_layers_params.append(weight.flatten())
                    layer.weight.data.zero_()
                if hasattr(layer, 'bias'):
                    bias = torch.zeros_like(layer.bias, device=self.device, dtype=torch.float)
                    head_layers_params.append(bias.flatten())
                    layer.bias.data.copy_(bias)

            flattened_params = torch.cat(base_layers_params + head_layers_params).unsqueeze(0)
            all_flattened_params.append(flattened_params)

        flattened_params_tensor = torch.cat(all_flattened_params, dim=0)
        return flattened_params_tensor

    def forward(self, x):
        x = x.float().to(self.device)
        x = self.base(x)
        x = self.head(x)
        return F.softmax(x, dim=1)

    def set_net_param(self, particles):
        particles = particles.to(self.device)
        cursor = 0
        for layer in self.base:
            if hasattr(layer, 'weight'):
                weight_size = layer.weight.numel()
                layer.weight.data = particles[cursor:cursor + weight_size].reshape(layer.weight.shape).float().requires_grad_(True)
                cursor += weight_size
            if hasattr(layer, 'bias'):
                bias_size = layer.bias.numel()
                bias_data = particles[cursor:cursor + bias_size].reshape(layer.bias.shape)
                layer.bias.data = bias_data.float().requires_grad_(True)
                cursor += bias_size

        for layer in self.head:
            if hasattr(layer, 'weight'):
                weight_size = layer.weight.numel()
                layer.weight.data = particles[cursor:cursor + weight_size].reshape(layer.weight.shape).float().requires_grad_(True)
                cursor += weight_size
            if hasattr(layer, 'bias'):
                bias_size = layer.bias.numel()
                bias_data = particles[cursor:cursor + bias_size].reshape(layer.bias.shape)
                layer.bias.data = bias_data.float().requires_grad_(True)
                cursor += bias_size


