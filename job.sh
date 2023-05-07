#!/bin/bash
#SBATCH -c 36
#SBATCH -w gnode071
#SBATCH --mem-per-cpu 2G
#SBATCH --gres gpu:4
#SBATCH --time 3-00:00:00
#SBATCH --output metrics.log
#SBATCH --mail-user sreejan.patel@students.iiit.ac.in
#SBATCH --mail-type ALL
#SBATCH --job-name evaluate

python3 evaluate.py --dataset "DHF1K" --weights /home2/sagarsj42/ViNetTest/ViNet/vinet_model.pt --criterion metrics
