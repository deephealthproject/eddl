#!/bin/bash

# Variables
EDDL_VERSION=0.4.3

# Install requirements
sudo chmod +x install_requirements.sh
sudo ./install_requirements.sh

# Download EDDL
wget https://github.com/deephealthproject/eddl/archive/$EDDL_VERSION.tar.gz
tar -xf $EDDL_VERSION.tar.gz

# Build EDDL
cd eddl-$EDDL_VERSION/
mkdir build
cd build/
cmake .. -DBUILD_PROTOBUF=ON -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON
make -j$(nproc)
make install

# Test EDDL
ctest --verbose

# Build docs (optional, check .dockerignore)
cd docs/doxygen/ && doxygen
cd ../source && make clean && make html
