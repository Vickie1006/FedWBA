#!/bin/bash

for RUN in 1 ; do

  python ./main.py --dataset FashionMNIST --num_svgd 60 --num_devices 50 --M 10 --k 10  --alpha_ada 0.0004 --epsilon_ada 0.00000001 --betta 0.9 

  python ./main.py --dataset FashionMNIST --num_svgd 60 --num_devices 100 --M 10 --k 20  --alpha_ada 0.0004 --epsilon_ada 0.00000001 --betta 0.9 

  python ./main.py --dataset FashionMNIST --num_svgd 60 --num_devices 200 --M 10 --k 40  --alpha_ada 0.0004 --epsilon_ada 0.00000001 --betta 0.9 

done

