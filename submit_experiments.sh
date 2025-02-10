#!/bin/bash

# Create experiments configuration file
cat > experiments.txt << EOL
random,10,0
random,20,0
random,50,0
structured,10,60
structured,20,60
structured,50,60
structured,10,180
structured,20,180
structured,50,180
EOL

# Submit array job
sbatch --array=1-9 train_array.sh