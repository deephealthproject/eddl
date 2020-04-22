#!/bin/bash

export CMAKE_LIBRARY_PATH=$PREFIX/lib:$PREFIX/include:$CMAKE_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$PREFIX

# MacOS build is simple, and will not be for CUDA
if [[ "$OSTYPE" == "darwin"* ]]; then
    MACOSX_DEPLOYMENT_TARGET=10.9
    CXX=clang++
    CC=clang
fi


# Build makefiles
mkdir build
cd build/
cmake -DBUILD_TARGET=CPU -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=$PREFIX $SRC_DIR

# Compile and install
make install -j8
