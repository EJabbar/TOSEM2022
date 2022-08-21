#!/bin/bash
#bash download_tools.sh
wget https://github.com/eclipse/org.aspectj/releases/download/V1_9_7/aspectj-1.9.7.jar -P ./$tools_folder/
wget https://repo1.maven.org/maven2/junit/junit/4.13.2/junit-4.13.2.jar -P ./$tools_folder/
wget https://repo1.maven.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar -P ./$tools_folder/
wget https://repo1.maven.org/maven2/org/mockito/mockito-core/3.12.4/mockito-core-3.12.4.jar -P ./$tools_folder/

unzip ./$tools_folder/aspectj-1.9.7.jar -d ./$tools_folder/aspectj
java -jar ./$tools_folder/aspectj-1.9.7.jar