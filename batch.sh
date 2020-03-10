#!/usr/bin/env bash
#SBATCH --array=0-170
#SBATCH --mem=64G
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1
#SBATCH --constraint="[gtx1080ti|p100|p6000]"
#SBATCH --time=2:00:00
#SBATCH --partition=batch
#SBATCH --output=logs/log-%x-slurm-%j.out
#SBATCH --error=errs/log-%x-slurm-%j.err
#SBATCH --job-name=GMR
#SBATCH --account=ghanembs
#SBATCH --qos=ghanembs-2020.03.12


~/miniconda3/envs/reg/bin/python main.py
