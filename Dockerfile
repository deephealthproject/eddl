FROM ubuntu:18.04

# Update software repository
RUN apt-get update
RUN apt-get install -y build-essential ca-certificates apt-utils # Essentials

# Get the latest repository
RUN apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common wget
RUN wget --no-check-certificate -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update

# Install dependencies
RUN apt-get install -y cmake graphviz zlib1g-dev # Utilities
RUN apt-get install -y libblas-dev liblapack-dev  # BLAS + LAPACK
RUN apt-get install -y libeigen3-dev  # Eigen3
#RUN apt-get install -y libprotobuf-dev protobuf-compiler protobuf-c-compiler  # Protobuf
RUN apt-get install -y libgtest-dev  # Google tests
RUN apt-get install -y autoconf automake libtool curl make g++ unzip git


# Install Protocol Buffers
RUN git clone https://github.com/protocolbuffers/protobuf.git && \
	cd protobuf && \
	git submodule update --init --recursive && \
    ./autogen.sh && \
	./configure && \
    make && \
    make check && \
    make install && \
    ldconfig 


RUN protoc --version



# Set working directory
ENV EDDL_ROOT /eddl
WORKDIR $EDDL_ROOT

# Copy repo
COPY . .

# All together
RUN mkdir build && \
    cd build && \
    cmake -D BUILD_SHARED_LIB=ON -D BUILD_EXAMPLES=ON .. && \
    make -j$(grep -c ^processor /proc/cpuinfo) && \
    make install


# Make build folder
#RUN mkdir build

# Set working directory
#WORKDIR $EDDL_ROOT/build

## Build EDDL
#RUN cmake .. -DBUILD_TARGET=CPU  # {CPU, GPU, FPGA}
#RUN make -j$(nproc)
