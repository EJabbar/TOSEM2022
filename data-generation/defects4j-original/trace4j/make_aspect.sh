#!/bin/bash

#bash make_aspect.sh

export PATH=$HOME/aspectj1.9/bin:$PATH
export CLASSPATH=$HOME/aspectj1.9/lib/aspectjrt.jar:$tools_folder/junit-4.13.2.jar

ajc -1.8 -outxml -outjar ./$tools_folder/aspect.jar Trace.aj