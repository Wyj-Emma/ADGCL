#!/bin/bash

#SBATCH --job-name=ADGCL
#SBATCH --partition=batch
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH -o %j
#SBATCH --error=%j.err
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end

source activate xxx(your virtual environment )
python ADGCL.py --dataset gowalla --epoch 180  --n_negs 10  --emb_reg 2.5e-5 --ssl_reg 0.1
python ADGCL.py --dataset amazon --epoch 180  --n_negs 5 --emb_reg 5e-5  --ssl_reg 0.2
python ADGCL.py --dataset tmall --epoch 180  --n_negs 10  --emb_reg 2.5e-5 --ssl_reg 0.2




