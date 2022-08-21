#!/bin/bash

#bash run.sh Codec 1b
project=$1
version=$2
i=$3

export tools_folder=tools
export root_folder=$(pwd)
export projects_folder=projects

path=$(echo "${project}_${version}_${i}" | awk '{print tolower($0)}')


bash d4j.sh $project $version ./$projects_folder/$path |& tee -a ./"${path}_script_log.txt"
bash log.sh ./$projects_folder/$path |& tee -a ./"${path}_script_log.txt"
