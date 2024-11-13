#!/bin/bash -e
#SBATCH --job-name=sleep_features   # job name (shows up in the queue)
#SBATCH --time=00-08:40:00  # Walltime (DD-HH:MM:SS)
#SBATCH --gpus-per-node=1   # GPU resources required per node
#SBATCH --cpus-per-task=4   # number of CPUs per task (1 by default)
#SBATCH --mem=128GB         # amount of memory per node (1 by default)
#SBATCH --account=aut03802  # Account

#Script to find the sleep features

module purge
module load TensorFlow/2.13.0-gimkl-2022a-Python-3.11.3

python attention_sleep.py
