#!/bin/bash

export CMAKE_LIBRARY_PATH=$PREFIX/lib:$PREFIX/include:$CMAKE_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$PREFIX

# Replacing the conda compilers is not a good idea, but if I don't this, it doesn't work
if [[ "$OSTYPE" == "darwin"* ]]; then
    MACOSX_DEPLOYMENT_TARGET=10.9
    CXX=clang++
    CC=clang

# If I don't do this, I get errors like: "undefined reference to `expf@GLIBC_2.27'"
elif [[ "$OSTYPE" == "linux"* ]]; then
    CXX=g++-7
    CC=gcc-7
fi

# Prints vars
echo "#################################################"
echo "##### CONDA BUILD CONSTANTS #####################"
echo "#################################################"
echo "PREFIX=$PREFIX"
echo "CMAKE_LIBRARY_PATH=$CMAKE_LIBRARY_PATH"
echo "CMAKE_PREFIX_PATH=$PREFIX"
echo "CMAKE_INSTALL_PREFIX=$PREFIX"
echo "SRC_DIR=$SRC_DIR"
echo "CC=$CC"
echo "CXX=$CXX"
echo "CPU_COUNT=$CPU_COUNT"
echo "#################################################"

# Build makefiles
mkdir build && cd build/ && cmake .. -DBUILD_TARGET=CPU \
      -DBUILD_SUPERBUILD=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_TESTS=OFF \
      -DCMAKE_C_COMPILER=$CC \
      -DCMAKE_CXX_COMPILER=$CXX \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_PREFIX_PATH=$PREFIX \
      -DCMAKE_INSTALL_PREFIX=$PREFIX \
      -DCMAKE_BUILD_TYPE=Release

# Compile
make -j${CPU_COUNT} ${VERBOSE_CM}

# Install
make install -j${CPU_COUNT}