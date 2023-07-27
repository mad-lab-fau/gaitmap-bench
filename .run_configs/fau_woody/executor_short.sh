#!/bin/bash

sbatch <<EOT
#!/bin/bash -l

#SBATCH --job-name="$2"
#SBATCH --time=03:00:00
#SBATCH --mail-user="$USER_EMAIL"
#SBATCH --mail-type=ALL
#SBATCH --output=../../results/hpc_logs/%j_$2.out
#SBATCH --ntasks=5

# load module
module load python/3.9-anaconda

# Make sure poetry is in path
export PATH="$HOME/.local/bin:$PATH"

echo "Starting job $2"
echo "Running command: $1"
eval "$1"
echo "Finished job $2"
EOT
