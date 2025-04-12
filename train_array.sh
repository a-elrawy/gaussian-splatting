#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24000M
#SBATCH --output=output/train_%A_%a.txt
#SBATCH --time=0-25:40            # time (DD-HH:MM)

# Equivalent salloc command for interactive allocation:
# salloc --gpus=1 --cpus-per-task=6 --mem=24G --time=00:40:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load cuda cudnn gcc python/3.10 opencv/4.10.0

source /home/elrawy/scratch/gs_env/bin/activate

nvidia-smi

# Read experiment configuration
CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments.txt)
SAMPLING_TYPE=$(echo $CONFIG | cut -d',' -f1)
NUM_VIEWS=$(echo $CONFIG | cut -d',' -f2)
ANGULAR_COVERAGE=$(echo $CONFIG | cut -d',' -f3)
RANGE_TYPE=$(echo $CONFIG | cut -d',' -f4)
# Name the experiment
EXPERIMENT_NAME="Sparse_${SAMPLING_TYPE}_${NUM_VIEWS}_${ANGULAR_COVERAGE}_${RANGE_TYPE}"
# Create a directory for the experiment
mkdir -p "experiments/$EXPERIMENT_NAME"

if [ -f "experiments/$EXPERIMENT_NAME/bicycle/results.json" ]; then
    echo "Experiment $EXPERIMENT_NAME has already been run. Skipping."
    exit
fi

echo "Configuration:"
echo "Sampling type: $SAMPLING_TYPE"
echo "Number of views: $NUM_VIEWS"
echo "Angular coverage: $ANGULAR_COVERAGE"
echo "Range type: $RANGE_TYPE"
echo "Experiment name: $EXPERIMENT_NAME"

# Run the training script with appropriate parameters
if [ "$SAMPLING_TYPE" = "random" ]; then
    python full_eval.py -m360 /home/elrawy/projects/def-emohamme/elrawy/SparseGS/gaussian-splatting/datasets/mipnerf360/ -tat /home/elrawy/projects/def-emohamme/elrawy/SparseGS/gaussian-splatting/datasets/tandt/ -db  /home/elrawy/projects/def-emohamme/elrawy/SparseGS/gaussian-splatting/datasets/db/ \
        --num_views $NUM_VIEWS \
        --sampling_type random \
        --exp_name $EXPERIMENT_NAME
else
    python full_eval.py -m360 /home/elrawy/projects/def-emohamme/elrawy/SparseGS/gaussian-splatting/datasets/mipnerf360/ -tat /home/elrawy/projects/def-emohamme/elrawy/SparseGS/gaussian-splatting/datasets/tandt/ -db  /home/elrawy/projects/def-emohamme/elrawy/SparseGS/gaussian-splatting/datasets/db/  \
        --num_views $NUM_VIEWS \
        --sampling_type structured \
        --angular_coverage $ANGULAR_COVERAGE \
        --range_type $RANGE_TYPE \
        --exp_name $EXPERIMENT_NAME
fi
