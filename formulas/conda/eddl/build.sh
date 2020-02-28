#!/bin/bash

export CMAKE_LIBRARY_PATH=$PREFIX/lib:$PREFIX/include:$CMAKE_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$PREFIX
export TH_BINARY_BUILD=1 # links CPU BLAS libraries thrice in a row (was needed for some MKL static linkage)

# MacOS build is simple, and will not be for CUDA
if [[ "$OSTYPE" == "darwin"* ]]; then
    MACOSX_DEPLOYMENT_TARGET=10.9
    CXX=clang++
    CC=clang
fi


# Build makefiles
cmake -DBUILD_TARGET=CPU -DBUILD_PROTOBUF=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=$PREFIX $SRC_DIR

# Compile and install
make install -j4
