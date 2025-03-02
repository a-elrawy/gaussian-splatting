#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24000M
#SBATCH --output=output/train_%A_%a.txt
#SBATCH --time=0-25:40            # time (DD-HH:MM)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load cuda cudnn gcc python/3.10 opencv/4.10.0

# Activate the environment
source /home/elrawy/projects/def-emohamme/elrawy/RobustGS/gs-env/bin/activate

# Install the requirements
cd /home/elrawy/projects/def-emohamme/elrawy/SparseGS/gaussian-splatting
# pip install -q submodules/diff-gaussian-rasterization
# pip install -q submodules/simple-knn

nvidia-smi

# Read experiment configuration
CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments.txt)
SAMPLING_TYPE=$(echo $CONFIG | cut -d',' -f1)
NUM_VIEWS=$(echo $CONFIG | cut -d',' -f2)
ANGULAR_COVERAGE=$(echo $CONFIG | cut -d',' -f3)
RANGE_TYPE=$(echo $CONFIG | cut -d',' -f4)
# Name the experiment
EXPERIMENT_NAME="${SAMPLING_TYPE}_${NUM_VIEWS}_${ANGULAR_COVERAGE}_${RANGE_TYPE}"
# Create a directory for the experiment
mkdir -p "experiments/$EXPERIMENT_NAME"

echo "Configuration:"
echo "Sampling type: $SAMPLING_TYPE"
echo "Number of views: $NUM_VIEWS"
echo "Angular coverage: $ANGULAR_COVERAGE"
echo "Range type: $RANGE_TYPE"
echo "Experiment name: $EXPERIMENT_NAME"

# Run the training script with appropriate parameters
if [ "$SAMPLING_TYPE" = "random" ]; then
    python full_eval.py -m360 datasets/mipnerf360/  -tat datasets/tandt/ -db  datasets/db/ \
        --num_views $NUM_VIEWS \
        --sampling_type random \
        --exp_name $EXPERIMENT_NAME
else
    python full_eval.py -m360 datasets/mipnerf360/ -tat datasets/tandt/ -db  datasets/db/  \
        --num_views $NUM_VIEWS \
        --sampling_type structured \
        --angular_coverage $ANGULAR_COVERAGE \
        --range_type $RANGE_TYPE \
        --exp_name $EXPERIMENT_NAME
fi
