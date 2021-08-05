#!/bin/bash

#SBATCH --job-name=cb_ca2a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=lg154@nyu.edu
#SBATCH --output=cb_ca2a.out
#SBATCH --gres=gpu # How much gpu need, n is the number

module purge
source ~/.bashrc
source /scratch/lg154/anaconda3/bin/activate
source activate python36
module load cuda/10.2.89
#module load cudnn/7.5

mkdir -p log
python train.py --fuse_type '2stage' --mmf_att 'CB_CA2a' --mode 'late' >log/train_cb_ca2a.log 2>& 1
echo "FINISH"
