#!/bin/bash

# Download source
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf/
git submodule update --init --recursive

# Build and install
./autogen.sh
./configure
make -j4
sudo make check -j4
sudo make install
ldconfig
