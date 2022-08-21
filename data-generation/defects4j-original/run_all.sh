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

# bash run.sh Cli 11b
# bash run.sh Cli 12b
# bash run.sh Cli 13b
# bash run.sh Cli 14b
# bash run.sh Cli 15b
# bash run.sh Cli 16b
# bash run.sh Cli 17b
# bash run.sh Cli 18b
# bash run.sh Cli 19b
# bash run.sh Cli 20b
# bash run.sh Cli 21b
# bash run.sh Cli 22b
# bash run.sh Cli 23b
# bash run.sh Cli 24b
# bash run.sh Cli 25b
# bash run.sh Cli 26b
# bash run.sh Cli 27b
# bash run.sh Cli 28b
# bash run.sh Cli 29b
bash run.sh Cli 30b
bash run.sh Cli 31b
bash run.sh Cli 32b
bash run.sh Cli 33b
bash run.sh Cli 34b



