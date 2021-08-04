#!/bin/bash

#SBATCH --job-name=2rs6_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=lg154@nyu.edu
#SBATCH --output=2rs6_2.out
#SBATCH --gres=gpu # How much gpu need, n is the number

module purge
source ~/.bashrc
source /scratch/lg154/anaconda3/bin/activate
source activate python36
module load cuda/10.2.89
#module load cudnn/7.5

mkdir -p log
python train.py --mmfs 'mmf=CA2b|act_fn=rsigmoid' --mrfs 'mrf=CA6|act_fn=tanh' --aux '2' >log/train_CA2b_rsigmoid_CA6_tanh_2.log 2>& 1
echo "FINISH"
