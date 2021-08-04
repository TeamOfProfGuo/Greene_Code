#!/bin/bash

#SBATCH --job-name=sig_Bct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=lg154@nyu.edu
#SBATCH --output=sig_Bct.out
#SBATCH --gres=gpu # How much gpu need, n is the number

module purge
source ~/.bashrc
source /scratch/lg154/anaconda3/bin/activate
source activate python36
module load cuda/10.2.89 
#module load cudnn/7.5

mkdir -p log
python train.py --mrf_att 'PA0' --mrf_act_fn 'sigmoid'  --mrf_fuse 'cat' --mrf_conv 'bblok' >log/train_pa0_sig_Bct.log 2>& 1
echo "FINISH"