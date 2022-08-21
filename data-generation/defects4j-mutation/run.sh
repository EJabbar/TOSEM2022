#!/bin/bash

project=$1
version=$2

modif_dir="./tmp/${project}_${version}_modif/"
ant_path=$(pwd)$(echo "/major/bin/")
javac_path=$(pwd)$(echo "/major/bin/javac")
#####################clone project################

mkdir -p ./trace4j/projects
mkdir -p ./tmp

defects4j checkout -p $project -v $version -w "./tmp/${project}_${version}_orig"
defects4j checkout -p $project -v $version -w $modif_dir

####################get number of tests###########

rm -f ./tmp/num_tests_${project}_${version}.txt

cd $modif_dir
defects4j compile
defects4j test

wc -l all_tests > ../num_tests_${project}_${version}.txt
rm -rf ./target
cd ../..

# ###################edit build file################

python change_xml.py $modif_dir $javac_path

# ####################mutant generation###########

cd $modif_dir
${ant_path}/ant compile
rm -rf ./target
cd ../..

python select_tests.py $modif_dir $project $version


# ####################replace mutation##############

python run_mutants.py "${modif_dir}/mutants/" $project $version








################################################################################################################################
# ####################create dataset##############

# mv ./trace4j/${lcproject}_${version}_* ./trace4j/logs/
# python filter_failing_tests.py $project $version

# rm -rf  ./trace4j/projects/${lcproject}_${version}_*