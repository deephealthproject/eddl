#!/usr/bin/bash
CPPFLAGS="-g -O3 -fopenmp -I ./eddl/src"
LDFLAGS="-g"
LDLIBS="-fopenmp -lpthread /usr/local/lib/libeddl.a -lz "
g++ $CPPFLAGS -c $1
g++ -o ${1%.cpp} $LDFLAGS ${1%.cpp}.o $LDLIBS
9