#!/bin/bash

# Create experiments configuration file
cat > experiments.txt << EOL
random,10,0,full
random,20,0,full
random,30,0,full
random,40,0,full
random,50,0,full
EOL

# Count number of experiments (excluding comment lines)
NUM_EXPS=$(grep -v '^#' experiments.txt | wc -l)
echo "Submitting $NUM_EXPS experiments..."

# Submit array job
sbatch --array=1-$NUM_EXPS train_array.sh

echo "Job array submitted. Check status with: sq"