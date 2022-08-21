#!/bin/bash

#bash log.sh ./temp/codec_1b
junit=$(pwd)/$tools_folder/junit-4.13.2.jar
hamcrest=$(pwd)/$tools_folder/hamcrest-core-1.3.jar
mockito=$(pwd)/$tools_folder/mockito-core-3.12.4.jar
aspectj=$(pwd)/$tools_folder/aspect.jar

cd ./$1;
defects4j_classpath=$(defects4j export -p cp.test)
export CLASSPATH=$junit:$hamcrest:$mockito:$aspectj:$defects4j_classpath

root_folder=trace4j
tests_statistic=$root_folder/test_statistic
logs=$root_folder/logs

mkdir -p ./$logs
mkdir -p ./$tests_statistic
# defects4j export -p tests.all > ./$tests_statistic/test_all.txt
python ../../filter_passings.py .

i=0

cat ./$tests_statistic/test_all.txt | (while read line
do

echo ""
echo "Number:${i}"
echo "File Name:${line}";
echo "";

i=$((i+1));
java -javaagent:../../$tools_folder/aspectj/files/lib/aspectjweaver.jar org.junit.runner.JUnitCore $line;

mv log.csv ./$logs/log_$line.csv;
mv ret_log.csv ./$logs/ret_log_$line.csv

done

echo ""
echo "Total Test Numbers:${i}")
