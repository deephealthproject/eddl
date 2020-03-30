!/bin/bash

echo "***********************************************************"
echo "**************** SCRIPTS FOR DEBIAN/UBUNTU ****************"
echo "***********************************************************"


# Install requirements ********************
echo "INSTALLING REQUIREMENTS ****************"
# Create temp folder
mkdir temp/
cd temp/

echo "Installing utilities ----------------------"
# Update software repository
sudo apt-get update
sudo apt-get install -y build-essential ca-certificates apt-utils # Essentials

# Install dependencies  ******************
# Utilities
sudo apt-get install -y cmake git wget graphviz zlib1g-dev libboost-all-dev

# Eigen3
echo "Installing Eigen3 ----------------------"
sudo apt-get install -y libeigen3-dev

# gTests
echo "Installing google tests ----------------------"
sudo apt-get install -y libgtest-dev
cd /usr/src/gtest
cmake CMakeLists.txt
make -j$(nproc)
cp *.a /usr/lib

# Protobuf
PROTOBUF_VERSION=3.11.4
echo "Installing protobuf $PROTOBUF_VERSION ----------------------"

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

# Install development libraries
echo "Installing development dependencies ----------------------"
sudo apt-get install -y doxygen
sudo apt-get install -y python3-pip
pip3 install sphinx
pip3 install sphinx_rtd_theme
pip3 install sphinx-tabs
pip3 install breathe
pip3 install pytest


# [A] Use specific version
EDDL_VERSION=0.4.4
echo "INSTALLING EDDL $EDDL_VERSION****************"

wget https://github.com/deephealthproject/eddl/archive/$EDDL_VERSION.tar.gz
tar -xf $EDDL_VERSION.tar.gz
cd eddl-$EDDL_VERSION/

# [B] Use this version
#cd ..

# Build EDDL
mkdir build
cd build/
#CUDA_TOOLKIT=PATH WHERE THE CUDA TOOLKIT IS INSTALLED
#CUDA_COMPILER=PATH WHERE THE CUDA COMPILER IS INSTALLED
cmake .. -DBUILD_PROTOBUF=ON -DBUILD_EXAMPLES=ON  # -DBUILD_TARGET=GPU -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT -DCMAKE_CUDA_COMPILER=$CUDA_COMPILER
make -j$(nproc)
make install

# Test EDDL
ctest --verbose

echo "BUILDING EDDL DOCUMENTATION***************"
# Build docs (optional, check .dockerignore)
cd ../docs/doxygen/ && doxygen
cd ../source && make clean && make html


# Install PyEDDL
cd ../../
PYEDDL_VERSION=0.6.0
echo "INSTALLING PYEDDL $PYEDDL_VERSION***************"
wget https://github.com/deephealthproject/pyeddl/archive/$PYEDDL_VERSION.tar.gz
tar -xf $PYEDDL_VERSION.tar.gz
cd pyeddl-$PYEDDL_VERSION/

export CPATH="/usr/include/eigen3:${CPATH}"
pip3 install numpy pybind11 pytest
python3 setup.py install
pytest tests
