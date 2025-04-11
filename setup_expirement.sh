#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24000M
#SBATCH --output=logs/%x-%j.out
#SBATCH --time=0-1:40            # time (DD-HH:MM)


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load cuda cudnn gcc python/3.10 opencv/4.10.0

# create a virtual environment if it does not exist
if [ ! -d /home/elrawy/scratch/gs_env ]; then
    python3 -m venv /home/elrawy/scratch/gs_env
fi

# activate the virtual environment
source /home/elrawy/scratch/gs_env/bin/activate
pip install --upgrade pip setuptools wheel

pip install torch torchvision torchaudio
pip install wandb tqdm plyfile
pip install --no-index 'numpy<2.0'

# Test torch installation
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"

# Install the requirements
cd /home/elrawy/scratch/work/gaussian-splatting
pip install -q submodules/diff-gaussian-rasterization
pip install -q submodules/simple-knn