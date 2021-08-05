#!/bin/bash

#SBATCH --job-name=ca6_003
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=lg154@nyu.edu
#SBATCH --output=ca6_003.out
#SBATCH --gres=gpu:1 # How much gpu need, n is the number

module purge
source ~/.bashrc
source /scratch/lg154/anaconda3/bin/activate
source activate python36
module load cuda/10.2.89 
#module load cudnn/7.5

mkdir -p log 
python train.py --mmf_att 'CA6' --lr '0.003' >log/train_ca6_003.log 2>& 1
echo "FINISH"
