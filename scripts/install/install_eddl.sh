#!/bin/bash

echo "***********************************************************"
echo "**************** SCRIPTS FOR DEBIAN/UBUNTU ****************"
echo "***********************************************************"

# Install requirements
sudo chmod +x install_requirements.sh
sudo ./install_requirements.sh

# [A] Use specific version
EDDL_VERSION=v0.5.4a
wget https://github.com/deephealthproject/eddl/archive/$EDDL_VERSION.tar.gz
tar -xf $EDDL_VERSION.tar.gz
cd eddl-$EDDL_VERSION/

# [B] Use this version
#cd ..

# Build EDDL
mkdir build
cd build/
cmake .. -DBUILD_SUPERBUILD=ON -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON
make -j$(nproc)
make install

# Test EDDL
bin/unit_tests

# Build docs (optional, check .dockerignore)
cd ../docs/doxygen/ && doxygen
cd ../sphinx/source && make clean && make html
