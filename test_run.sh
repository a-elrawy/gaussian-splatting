#!/bin/bash
#SBATCH --gpus-per-node=1         # Number of GPU(s) per node
#SBATCH --cpus-per-task=6         # CPU cores/threads
#SBATCH --mem=24000M               # memory per node
#SBATCH --output=output/train-%j.txt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load cuda
module load cudnn
module load gcc
module load python/3.10
module load opencv/4.9.0

# Activate the environment
source /home/elrawy/projects/def-emohamme/elrawy/RobustGS/gs-env/bin/activate

# Install the requirements
cd /home/elrawy/projects/def-emohamme/elrawy/SparseGS/gaussian-splatting

pip install -q submodules/diff-gaussian-rasterization
pip install -q submodules/simple-knn


nvidia-smi
# Print 


# Run the script
python train.py -s tandt/train --eval