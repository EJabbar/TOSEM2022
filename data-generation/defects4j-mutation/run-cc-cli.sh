#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --account=def-hemmati-ab
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=48G
#SBATCH --job-name=log_d4j
#SBATCH --output=%x-%j.out

####### Email
#SBATCH --mail-user=emad.jabbar@ucalgary.ca
#SBATCH --mail-type=ALL

bash run-compute-canada.sh Cli 39f
# bash run-compute-canada.sh Cli 35f
# bash run-compute-canada.sh Cli 32f
# bash run-compute-canada.sh Cli 33f
