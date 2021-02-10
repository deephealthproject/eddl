#!/usr/bin/env bash

# Build and install
sudo apt-get install -y libgtest-dev
cd /usr/src/gtest
cmake CMakeLists.txt
make -j$(nproc)
cp *.a /usr/lib