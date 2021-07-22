#!/bin/bash

#SBATCH --job-name=rfu_pa7a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=88GB
#SBATCH --time=50:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=lg154@nyu.edu
#SBATCH --output=rfu_pa7a.out
#SBATCH --gres=gpu:2 # How much gpu need, n is the number
#SBATCH -p aquila
#SBATCH --constraint=2080Ti

module purge
source ~/.bashrc
source activate python36
module load cuda/10.0
#module load cudnn/7.5

python train.py --mmf_att 'PA7a' >log/train_pa7a.log 2>& 1
echo "FINISH"

