#!/bin/bash

for RUN in 1 ; do

  python ./main.py --dataset CIFAR10 --num_svgd 75 --num_devices 50 --M 10 --k 10  --alpha_ada 0.0002 --epsilon_ada 0.0000000001 --betta 0.9

  python ./main.py --dataset CIFAR10 --num_svgd 75 --num_devices 100 --M 10 --k 20  --alpha_ada 0.0002 --epsilon_ada 0.0000000001 --betta 0.9

  python ./main.py --dataset CIFAR10 --num_svgd 75 --num_devices 200 --M 10 --k 40  --alpha_ada 0.0002 --epsilon_ada 0.0000000001 --betta 0.9

done

