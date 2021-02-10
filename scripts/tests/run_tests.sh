#!/usr/bin/env bash

echo "***********************************************************"
echo "************************ RUN TESTS ************************"
echo "***********************************************************"

# Download source code
git clone https://github.com/deephealthproject/eddl.git
cd eddl/

# Set source and building path
SOURCE_PATH=$(pwd)
BUILD_PATH=build

# Compile for CPU ##########################
# Remove and install dependencies
conda remove --name eddl --all
conda env create -f environment-cpu.yml
conda activate eddl

# Remove and create build path
rm -R $BUILD_PATH
mkdir $BUILD_PATH

# Generate CPU makefiles
cmake -H${SOURCE_PATH} -B${BUILD_PATH} -DBUILD_TARGET=CUDNN  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX  -DCMAKE_CXX_COMPILER=/usr/bin/g++-7

# Compile
(cd $BUILD_PATH && make -j$(nproc))

# Run examples
 ./run_examples cpu 5  # build_target num_epochs

# Compile for GPU ##########################
# Remove and install dependencies
conda remove --name eddl --all
conda env create -f environment-cpu.yml
conda activate eddl

# Remove and create build path
rm -R $BUILD_PATH
mkdir $BUILD_PATH

# Generate GPU makefiles
cmake -H${SOURCE_PATH} -B${BUILD_PATH} -DBUILD_TARGET=GPU  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CXX_COMPILER=/usr/bin/g++-7

# Compile
(cd $BUILD_PATH && make -j$(nproc))

# Run examples
 ./run_examples gpu 5  # build_target num_epochs

# Compile for CUDNN ##########################
# Remove and install dependencies
conda remove --name eddl --all
conda env create -f environment-cpu.yml
conda activate eddl

# Remove and create build path
rm -R $BUILD_PATH
mkdir $BUILD_PATH

# Generate CUDNN makefiles
cmake -H${SOURCE_PATH} -B${BUILD_PATH}  -DBUILD_TARGET=CUDNN  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CXX_COMPILER=/usr/bin/g++-7

# Compile
(cd $BUILD_PATH && make -j$(nproc))

# Run examples
 ./run_examples cudnn 5  # build_target num_epochs