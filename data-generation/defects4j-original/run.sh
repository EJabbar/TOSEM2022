#!/bin/bash

project=$1
version=$2

#####################clone project################

cd ./trace4j/

bash ./run.sh $project $version

cd ../
#####################extract tests##############
lcproject=$(echo ${project,,})

mkdir -p ./dataset/${lcproject}_${version}
python ./extract_tests.py $project $version

# rm -rf  ./trace4j/projects/${lcproject}_${version}