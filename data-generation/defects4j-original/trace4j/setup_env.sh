#!/bin/bash
#bash setup_env.sh

bash check_env.sh

export tools_folder=tools
mkdir -p $tools_folder/aspectj

export root_folder=$(pwd)
bash download_tools.sh
bash make_aspect.sh