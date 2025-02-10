#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24000M
#SBATCH --output=output/train_%A_%a.txt

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load cuda cudnn gcc python/3.10 opencv/4.9.0

# Activate the environment
source /home/elrawy/projects/def-emohamme/elrawy/RobustGS/gs-env/bin/activate

# Install the requirements
cd /home/elrawy/projects/def-emohamme/elrawy/SparseGS/gaussian-splatting
pip install -q submodules/diff-gaussian-rasterization
pip install -q submodules/simple-knn

nvidia-smi

# Read experiment configuration
CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments.txt)
SAMPLING_TYPE=$(echo $CONFIG | cut -d',' -f1)
NUM_VIEWS=$(echo $CONFIG | cut -d',' -f2)
ANGULAR_COVERAGE=$(echo $CONFIG | cut -d',' -f3)

# Set output directory based on configuration
if [ "$SAMPLING_TYPE" = "random" ]; then
    OUTPUT_DIR="output/random_${NUM_VIEWS}views"
else
    OUTPUT_DIR="output/structured_${NUM_VIEWS}views_${ANGULAR_COVERAGE}deg"
fi

# Run the training script with appropriate parameters
if [ "$SAMPLING_TYPE" = "random" ]; then
    python train.py -s tandt/train \
        --num_views $NUM_VIEWS \
        --sampling_type random \
        --eval \
        --model_path $OUTPUT_DIR
else
    python train.py -s tandt/train \
        --num_views $NUM_VIEWS \
        --sampling_type structured \
        --angular_coverage $ANGULAR_COVERAGE \
        --eval \
        --model_path $OUTPUT_DIR
fi