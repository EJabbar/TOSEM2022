#!/bin/bash

project=$1
version=$2

python --version

echo "Start Time:" $(date -u)

bash run.sh $project $version

echo "END Time:" $(date -u)