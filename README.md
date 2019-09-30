<p align="center">
  <img src="https://raw.githubusercontent.com/salvacarrion/salvacarrion.github.io/master/assets/hot-linking/logo-eddl.png" alt="EDDL" height="140" width="300">
</p>

-----------------

**EDDL** is an open source library for numerical computation tailored to the healthcare domain.
> More information: [https://deephealth-project.eu/](https://deephealth-project.eu/)


# Requirements

- CMake 3.9.2 or higher
- A modern compiler with C++11 support

To clone all third_party submodules use:

```bash
git clone --recurse-submodules -j8 https://github.com/deephealthproject/eddl.git
```


# Installation

To build `eddl`, clone or download this repository and then, from within the repository, run:

```bash
mkdir build
cd build
cmake ..
make
```

Compiler flags and options:

- `-DBUILD_PYTHON=ON`: Compiles Python binding
- `-DBUILD_TESTS=ON`: Compiles tests
- `-DBUILD_EXAMPLES=ON`: Compiles examples
- `-DBUILD_TARGET=cpu`: Compiles for {`cpu`, `gpu` or `fpga`}


# Windows specific installation

Default for `Visual Studio 15 2017` build envrionment is x86, while EDDLL requires x64. This can be changed by typing `cmake -A x64 .` as cmake command.

On Windows, the POSIX threads library is required. Path to this library can be specified to cmake as follows: `env PTHREADS_ROOT=path_to_pthreads cmake -A x64 .`
The PThreads library can be found at [https://sourceforge.net/projects/pthreads4w/](https://sourceforge.net/projects/pthreads4w/).


# Tests

To execute all unit tests, go to your build folder and run the following command:

```bash
make test
```


# Getting started

```c++
#include "apis/eddl.h"

using namespace eddl;

int main(int argc, char **argv) {

    // Download dataset
    download_mnist();

    // Settings
    int epochs = 10;
    int batch_size = 1000;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var
    l = Activation(Dense(l, 1024), "relu");
    l = Activation(Dense(l, 1024), "relu");
    l = Activation(Dense(l, 1024), "relu");
    layer out = Activation(Dense(l, num_classes), "softmax");
    model net = Model({in}, {out});

    // View model
    summary(net);
    plot(net, "model.pdf");

    // Build model
    build(net,
          sgd(0.01, 0.9), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_CPU(4) // CPU with 4 threads
    );

    // Load dataset
    tensor x_train = T_load("trX.bin");
    tensor y_train = T_load("trY.bin");
    tensor x_test = T_load("tsX.bin");
    tensor y_test = T_load("tsY.bin");

    // Preprocessing
    div_(x_train, 255.0);
    div_(x_test, 255.0);

    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);

    // Evaluate test
    std::cout << "Evaluate test:" << std::endl;
    evaluate(net, {x_test}, {y_test});
}

```

You can find more examples in the _examples_ folder.


# Continuous build status

| System  |  Compiler  | Status |
|:-------:|:----------:|:------:|
| Windows (CPU) | VS 15.9.11 | [![Build Status](https://jenkins-master-deephealth-unix01.ing.unimore.it/badge/job/DeepHealth/job/eddl/job/master/windows_end?)](https://jenkins-master-deephealth-unix01.ing.unimore.it/job/DeepHealth/job/eddl/job/master/)      |
| Linux (CPU)   | GCC 5.5.0  | [![Build Status](https://jenkins-master-deephealth-unix01.ing.unimore.it/badge/job/DeepHealth/job/eddl/job/master/linux_end?)](https://jenkins-master-deephealth-unix01.ing.unimore.it/job/DeepHealth/job/eddl/job/master/)        |
| Windows (GPU) | VS 15.9.11 | [![Build Status](https://jenkins-master-deephealth-unix01.ing.unimore.it/badge/job/DeepHealth/job/eddl/job/master/windows_gpu_end?)](https://jenkins-master-deephealth-unix01.ing.unimore.it/job/DeepHealth/job/eddl/job/master/)  |
| Linux (GPU)   | GCC 5.5.0  | [![Build Status](https://jenkins-master-deephealth-unix01.ing.unimore.it/badge/job/DeepHealth/job/eddl/job/master/linux_gpu_end?)](https://jenkins-master-deephealth-unix01.ing.unimore.it/job/DeepHealth/job/eddl/job/master/)    |

Documentation available [here](http://imagelab.ing.unimore.it/eddl/).


# Python wrapper

If you are not a C++ fan, try [PyEDDL](https://github.com/deephealthproject/pyeddl), a python wrapper for this library.

# FAQs

- **When I run an example from `examples/` I get `segmentation fault (core dumped)`**: 
    - **CPU**: This is probably because your processor does not support 
    AVX instructions. Try to compile the source with the optimization flags: `OPT=2` or `OPT=3` (uppercase). 
    - **GPU**: Make sure you are using the computing service: `CS_GPU`.