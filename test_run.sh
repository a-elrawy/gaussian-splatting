#!/bin/bash
#SBATCH --gpus-per-node=1         # Number of GPU(s) per node
#SBATCH --cpus-per-task=6         # CPU cores/threads
#SBATCH --mem=24000M               # memory per node
#SBATCH --output=output/train-%j.txt
#SBATCH --time=0-2:30            # time (DD-HH:MM)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load cuda
module load cudnn
module load gcc
module load python/3.10
module load opencv/4.10.0

# Activate the environment
source /home/elrawy/projects/def-emohamme/elrawy/RobustGS/gs-env/bin/activate

cd /home/elrawy/projects/def-emohamme/elrawy/SparseGS/gaussian-splatting
# Install the requirements

# pip install -q submodules/diff-gaussian-rasterization
# pip install -q submodules/simple-knn


nvidia-smi


# Run the script
python train.py -s datasets/tandt/truck \
    --num_views 30 \
    --sampling_type structured \
    --angular_coverage 60 \
    --eval 