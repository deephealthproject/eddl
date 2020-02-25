#!/bin/bash

cmake -DBUILD_PROTOBUF=OFF -DCMAKE_INSTALL_PREFIX=$PREFIX $SRC_DIR
make install -j
