#!/bin/bash

# Create temp folder
mkdir temp/
cd temp/

# Update software repository
sudo apt-get update
sudo apt-get install -y build-essential ca-certificates apt-utils # Essentials

# Install dependencies  ******************
# Utilities
sudo apt-get install -y cmake git wget graphviz zlib1g-dev libboost-all-dev

# Eigen3
sudo apt-get install -y libeigen3-dev

# gTests
sudo chmod +x install_gTests.sh
sudo ./install_gTests.sh

# Protobuf
sudo chmod +x install_protobuf.sh
sudo ./install_protobuf.sh

# Install development libraries
sudo apt-get install -y doxygen
sudo apt-get install -y python3-pip
pip3 install sphinx
pip3 install sphinx_rtd_theme
pip3 install sphinx-tabs
pip3 install breathe
