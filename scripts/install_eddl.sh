#!/bin/bash

echo "***********************************************************"
echo "**************** SCRIPTS FOR DEBIAN/UBUNTU ****************"
echo "***********************************************************"

# Install requirements
sudo chmod +x install_requirements.sh
sudo ./install_requirements.sh

# [A] Use specific version
EDDL_VERSION=0.4.4
wget https://github.com/deephealthproject/eddl/archive/$EDDL_VERSION.tar.gz
tar -xf $EDDL_VERSION.tar.gz
cd eddl-$EDDL_VERSION/

# [B] Use this version
#cd ..

# Build EDDL
mkdir build
cd build/
cmake .. -DBUILD_PROTOBUF=ON -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON
make -j$(nproc)
make install

# Test EDDL
ctest --verbose

# Build docs (optional, check .dockerignore)
cd ../docs/doxygen/ && doxygen
cd ../source && make clean && make html
