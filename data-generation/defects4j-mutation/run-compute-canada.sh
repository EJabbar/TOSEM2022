#!/bin/bash

project=$1
version=$2

module load nixpkgs/16.09 java/1.8.0_192
export JAVA_TOOL_OPTIONS="-Xmx16g"
source /home/ejabbar/MutENV/bin/activate

python --version

echo "Start Time:" $(date -u)
bash run.sh $project $version

echo "END Time:" $(date -u)