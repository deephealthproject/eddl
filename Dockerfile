FROM ubuntu:20.04

# Install minimum dependencies  ******************
RUN apt-get update
RUN apt-get install -y build-essential ca-certificates apt-utils checkinstall # Essentials
RUN apt-get install -y git wget vim

# Install miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Activate conda
ENV PATH="/root/miniconda3/bin:$PATH"
RUN conda --version

# Set working directory
WORKDIR /eddl

# Environment first (to reduce the building time if something has change)
COPY environment.yml .

# Install dependencies
RUN conda update conda && \
    conda env create -f environment.yml

# Copy repo
COPY . .

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "eddl", "/bin/bash", "-c"]

# Build EDDL
RUN mkdir build
RUN cd build && \
    cmake .. \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DBUILD_SUPERBUILD=OFF \
    -DBUILD_TARGET=CPU \
    -DBUILD_HPC=OFF -DBUILD_TESTS=ON \
    -DBUILD_DIST=OFF -DBUILD_RUNTIME=OFF
RUN cd build && \
    make -j$(nproc) && \
    make install

# Test EDDL
RUN cd build/bin/ && ./unit_tests

# Build docs (optional, check .dockerignore)
RUN cd docs/doxygen/ && doxygen
RUN cd docs/sphinx/source/ && make clean && make html

