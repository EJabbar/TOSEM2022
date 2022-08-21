#!/bin/bash

#bash run_computecanada.sh Codec b 1 16

####### Properties
#SBATCH --time=06:00:00
#SBATCH --account=def-hemmati-ab
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=48G
#SBATCH --job-name=log_d4j
#SBATCH --output=%x-%j.out

####### Email
#SBATCH --mail-user=ehsan.mashhadi@gmail.com
#SBATCH --mail-type=ALL

PROJECT=$1
VERSION=$2
START=$3
END=$4

echo "Start Time:" $(date -u)

module load java/1.8.0_192
export JAVA_TOOL_OPTIONS="-Xmx16g"
for i in $(seq $START $END);
 do (bash run.sh $PROJECT "${i}${VERSION}");
done

echo "END Time:" $(date -u)

