#!/usr/bin/env bash

# Variables
PROTOBUF_VERSION=3.11.4

# Install requirements
sudo apt-get install -y wget
sudo apt-get install -y autoconf automake libtool curl make g++ unzip

# Download source
wget https://github.com/protocolbuffers/protobuf/releases/download/v$PROTOBUF_VERSION/protobuf-cpp-$PROTOBUF_VERSION.tar.gz
tar -xf protobuf-cpp-$PROTOBUF_VERSION.tar.gz

# Build and install
cd protobuf-$PROTOBUF_VERSION
./configure
make -j$(nproc)
make install
ldconfig