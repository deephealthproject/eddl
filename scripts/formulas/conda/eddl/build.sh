#!/bin/bash

# MacOS build is simple, and will not be for CUDA
if [[ "$OSTYPE" == "darwin"* ]]; then
    MACOSX_DEPLOYMENT_TARGET=10.9 \
        CXX=clang++ \
        CC=clang
fi

cmake -DBUILD_PROTOBUF=ON -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=$PREFIX $SRC_DIR
make install -j4
