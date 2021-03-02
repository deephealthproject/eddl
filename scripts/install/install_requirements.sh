#!/usr/bin/env bash

# Create temp folder
mkdir temp/
cd temp/

# Update software repository
sudo apt-get update
sudo apt-get install -y build-essential ca-certificates apt-utils # Essentials

# Install dependencies  ******************
# Utilities
sudo apt-get install -y cmake git wget graphviz libeigen3-dev zlib1g-dev

# Install development libraries
sudo apt-get install -y doxygen
sudo apt-get install -y python3-pip
pip3 install sphinx
pip3 install sphinx_rtd_theme
pip3 install sphinx-tabs
pip3 install breathe
pip3 install pytest

