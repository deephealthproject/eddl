# Installation

If you are installing from source, you will need a C++11 compiler. Also, we highly recommend installing an [Anaconda environment](https://docs.conda.io/en/latest/miniconda.html). 
You will get a high-quality BLAS library (MKL) and you get controlled dependency versions regardless of your Linux distro.

If you want to compile with CUDA support, install

- NVIDIA CUDA 9 or above
- NVIDIA cuDNN v7 or above

Once you have [Anaconda](https://docs.conda.io/en/latest/miniconda.html) installed, here are the instructions.


## Download source code

To get the source, download one of the release .tar.gz or .zip packages in the release page:

```bash
git clone https://github.com/deephealthproject/eddl.git
```


## Prerequisites

To build EDDL from source, the following tools are needed:

- C++11-standard-compliant compiler
- graphviz
- wget
- cmake
- eigen
- cudatoolkit
- zlib
- gtest
- protobuf
- libprotobuf
- doxygen
- python
- pip:
    - sphinx
    - breathe
    - sphinx_rtd_theme
    - sphinx-tabs

These dependencies can be installed either manually or using a conda package manager (recommended).


### Anaconda package manager (recommended)

The required libraries are easier to install if you use using a [anaconda package manager](https://docs.conda.io/en/latest/miniconda.html)).
Once conda is installed in your system, you can use the `environment.yml` file inside the `eddl/`folder to install the requirements.

To create and activate the conda environment use the following commands:

```bash
conda env create -f environment.yml
conda activate eddl
```

> Note:
> If the conda environment misses some dependency, please write an issue and complete the installation manually


### Manual management of dependencies

Regardless of your platform, install:

- CUDA: https://developer.nvidia.com/cuda-toolkit
- Google benchmark: https://github.com/google/benchmark
- Protobuf: https://github.com/protocolbuffers/protobuf/blob/master/src/README.md

Then, on Ubuntu/Debian install:

```
sudo apt-get install build-essential git graphviz wget zlib1g-dev cmake  # Utilities
sudo apt-get install libblas-dev liblapack-dev  # BLAS + LAPACK
sudo apt-get install libeigen3-dev  # Eigen3
sudo apt-get install libgtest-dev  # Google tests
sudo apt-get install libboost-all-dev
```

Or, on MacOS install:

```
brew install git graphviz wget zlib cmake  # Utilities
brew install openblas lapack # BLAS + LAPACK
brew install eigen
# Install Google Tests: https://github.com/google/googletest
brew install boost
```


### Docker image

You will need a [docker engine](https://docs.docker.com/install/)

To build the EDDL from the docker image, go to the `eddl` folder and run:

```
docker build -t eddl .
```

Then, you can execute this line to launch a shell in the image:

```
docker run -it eddl /bin/bash
```

Or mount it, if you want to **edit the code** from the host machine:

```
docker run -it -v $(pwd):/eddl/ eddl /bin/bash
```



## Compilation

To build and install the EDDL library from source, execute the following commands inside the `eddl/` folder:

```bash
mkdir build
cd build
cmake .. -DBUILD_TARGET=CPU  # {CPU, GPU, FPGA}
make -j 4  # num_cores
make install
```

> Note:
> To known the number of logical cores type: `nproc` (linux) or `sysctl -n hw.logicalcpu` (mac os)


### Building flags

#### Backend support

**CPU support:**
By default the EDDL is build for CPU. If you have any problem and want to compile for CPU, try adding `BUILD_TARGET=CPU` to your cmake options.

```bash
-DBUILD_TARGET=CPU
```

**GPU (CUDA) support:**
If you have CUDA installed, you can build EDDL with GPU support by adding `BUILD_TARGET=GPU` to your cmake options.

```bash
-DBUILD_TARGET=GPU
```

**FPGA support:**
If available, you can build EDDL with FPGA support by adding `BUILD_TARGET=FPGA` to your cmake options.

```bash
-DBUILD_TARGET=FPGA
```

> Not yet implemented


#### Additional flags

These flags can enable additional features of the EDDL or help you troubleshooting the installation.

**C++ compiler::**
If you have problems with the default g++ compiler, try setting `EIGEN3_INCLUDE_DIR`, such as:

```bash
-DCMAKE_CXX_COMPILER=/path/to/c++compiler
```

**Eigen3:**
At the core of many numerical operations, we use [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page).
If CMake is unable to find Eigen3 automatically, try setting `Eigen3_DIR`, such as:

```bash
-DEigen3_DIR=/path/to/eigen
```

**Intel MKL:**
EDDL can leverage Intel's MKL library to speed up computation on the CPU.

To use MKL, include the following cmake option:

```bash
-DMKL=TRUE
```

If CMake is unable to find MKL automatically, try setting MKL_ROOT, such as:

```bash
-DMKL_ROOT="/path/to/MKL"
```

**CUDA:**
If CMake is unable to find CUDA automatically, try setting `CUDA_TOOLKIT_ROOT_DIR`, such as:

```bash
-DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda
```

**Build examples:**
To compile the examples, use the setting `BUILD_EXAMPLES`, such as:

```bash
-DBUILD_EXAMPLES=ON
```

> Notes: The examples can be found in `build/targets/`


**Build tests:**
To compile the tests, use the setting `BUILD_TESTS`, such as:

```bash
-DBUILD_TESTS=ON
```

**Build shared library:**
To compile the EDDL as a shared library, use the setting `BUILD_SHARED_LIB`, such as:

```bash
-DBUILD_SHARED_LIB=ON
```


### Windows specific installation

Default for `Visual Studio 15 2017` build environment is x86, while EDDL requires x64. This can be changed by typing `cmake -A x64 .` as cmake command.

On Windows, the POSIX threads library is required. Path to this library can be specified to cmake as follows: `env PTHREADS_ROOT=path_to_pthreads cmake -A x64 .`
The PThreads library can be found at [https://sourceforge.net/projects/pthreads4w/](https://sourceforge.net/projects/pthreads4w/).


## FAQs

- **When I run an example from `examples/` I get `segmentation fault (core dumped)`**:
    - **CPU**: This is probably because your processor does not support
    AVX instructions. Try to compile the source with the optimization flags: `OPT=2` or `OPT=3` (uppercase).
    - **GPU**: Make sure you are using the computing service: `CS_GPU`.
- **Protobuf doesn't work/compilation error(temporal fix)**:
    1) Make sure you have `protbuf` and `libprotobuf` installed
    2) Go to `src/serialization/onnx/` and delete these files: `onnx.pb.cc` and `onnx.pb.cc`.
    3) Rebuild them using `protoc --cpp_out=. onnx.proto` (you need to be at `src/serialization/onnx/`)
