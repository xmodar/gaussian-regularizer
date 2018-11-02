#!/bin/bash
#SBATCH --job-name VGG_GNM
#SBATCH --array=0-134
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --mem 10GB
#SBATCH --reservation=IVUL

module purge
module load applications-extra
module load cudnn/7.0.3-cuda9.0.176
module load anaconda/2.1.0
source activate base

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $DIR

python main.py file -e $SLURM_ARRAY_TASK_ID