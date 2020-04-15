FROM ubuntu:18.04

# Update software repository
RUN apt-get update
RUN apt-get install -y build-essential ca-certificates apt-utils # Essentials

# Install dependencies  ******************
# Utilities
RUN apt-get install -y cmake git wget graphviz

# Install development libraries
RUN apt-get install -y doxygen
RUN apt-get install -y python3-pip
RUN pip3 install sphinx
RUN pip3 install sphinx_rtd_theme
RUN pip3 install sphinx-tabs
RUN pip3 install breathe
# ****************** Install dependencies

# Set working directory
ENV EDDL_ROOT /eddl
WORKDIR $EDDL_ROOT

# Copy repo
COPY .. .

# Build EDDL
RUN mkdir build
RUN cd build && \
    cmake .. \
    make -j$(nproc) && \
    make install

# Test EDDL
RUN cd build && cmake/eddl/bin/unit_tests

# Build docs (optional, check .dockerignore)
RUN cd docs/doxygen && doxygen
RUN cd docs/source && make clean && make html

