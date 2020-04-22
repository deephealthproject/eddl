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
sudo apt-get install -y cmake git wget graphviz libeigen3-dev zlib1g-dev libboost-all-dev

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
EDDL_VERSION=v0.5a
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
cmake .. -DBUILD_EXAMPLES=ON  # -DBUILD_TARGET=GPU -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT -DCMAKE_CUDA_COMPILER=$CUDA_COMPILER
make -j$(nproc)
make install

# Test EDDL
bin/unit_tests

echo "BUILDING EDDL DOCUMENTATION***************"
# Build docs (optional, check .dockerignore)
cd ../docs/doxygen/ && doxygen
cd ../sphinx/source && make clean && make html


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
