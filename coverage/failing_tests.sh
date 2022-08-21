#!/bin/bash

pname=$1
version=$2


cd ./${pname}_${version}
defects4j export -p tests.trigger > ../failing_tests/${pname}/failing_${version}.txt
cd ..
