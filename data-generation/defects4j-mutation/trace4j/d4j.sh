#!/bin/bash
#bash d4j Codec 1b ./temp/codec_1b/
cd $3
defects4j compile
defects4j test
