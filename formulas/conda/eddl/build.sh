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
    CXX=g++
    CC=gcc
fi


# Build makefiles
mkdir build
cd build/
cmake -DBUILD_TARGET=GPU \
      -DBUILD_SUPERBUILD=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=$PREFIX \
      $SRC_DIR

# Compile
make -j${CPU_COUNT} ${VERBOSE_CM}

# Install
make install -j${CPU_COUNT}