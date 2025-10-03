## Personalized Bayesian Federated Learning with Wasserstein Barycenter Aggregation

This repository hosts the official implementation of the Personalized Bayesian Federated Learning framework leveraging Wasserstein Barycenter Aggregation (dubbed FedWBA).

## Abstract

Personalized Bayesian federated learning (PBFL) handles non-i.i.d. client data and quantifies uncertainty by combining personalization with Bayesian inference. However, existing PBFL methods face two limitations: restrictive parametric assumptions in client posterior inference and naive parameter averaging for server aggregation.
To overcome these issues, we propose FedWBA, a novel PBFL method that enhances both local inference and global aggregation. At the client level, we use particle-based variational inference for nonparametric posterior representation. At the server level, we introduce particle-based Wasserstein barycenter aggregation, offering a more geometrically meaningful approach.  
Theoretically, we provide local and global convergence guarantees for FedWBA. Locally, we prove a KL divergence decrease lower bound per iteration for variational inference convergence. Globally, we show that the Wasserstein barycenter converges to the true parameter as the client data size increases. Empirically, experiments show that FedWBA outperforms baselines in prediction accuracy, uncertainty calibration, and convergence rate, with ablation studies confirming its robustness.

## Downloading dependencies

```
pip install -r requirements.txt  
```

## Experiments

All commands for running the paper experiments can be found in ./experiments.
According to the following code, the results of MNIST on 50 clients can be reproduced. For 100 clients, set k to 20.




