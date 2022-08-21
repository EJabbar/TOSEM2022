#!/bin/bash

pname=$1
version=$2


cd ./${pname}_${version}
defects4j coverage -r
cp all_tests ../relevant_tests/${pname}/relevant_${version}.txt
cd ..
