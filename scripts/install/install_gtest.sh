#!/usr/bin/env bash

# Install dependencies
sudo apt-get update
sudo apt-get wget

# Build and install
wget https://github.com/google/googletest/archive/release-1.10.0.tar.gz
tar -xvzf googletest-release-1.10.0.tar.gz
cd googletest-release-1.10.0
mkdir build
cd build && cmake .. && make -j$(nproc) install

# Build and install
#sudo apt-get install -y libgtest-dev
#cd /usr/src/gtest
#cmake CMakeLists.txt
#make -j$(nproc)
#cp *.a /usr/lib