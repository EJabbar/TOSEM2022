#!/bin/bash
#bash d4j Codec 1b ./temp/codec_1b/
mkdir -p $3
defects4j checkout -p $1 -v $2 -w $3
cd $3
defects4j compile
defects4j test
