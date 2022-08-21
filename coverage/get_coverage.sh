#!/bin/bash

id=$1
tid=$2
name=$3
pname=$4

cd ./${pname}_${id}

defects4j coverage -t ${name} > coverage.txt
cp coverage.xml ../coverage_results/${pname}/coverage_${id}_${tid}.xml
cp coverage.txt ../coverage_results/${pname}/coverage_${id}_${tid}.txt
cd ..