# European Distributed Deep Learning Library

<div align="center">
  <img src="https://raw.githubusercontent.com/salvacarrion/salvacarrion.github.io/master/assets/hot-linking/logo-eddl.png">
</div>

-----------------

**EDDL** is European Deep Learning Library for numerical computation tailored to the healthcare domain.
> More information: [https://deephealth-project.eu/](https://deephealth-project.eu/)


# Requirements

- CMake 3.9.2 or higher
- A modern compiler with C++11 support

To clone all third_party submodules use:

```bash
 `git clone --recurse-submodules -j8 https://github.com/deephealthproject/eddl.git`
 ```


# Installation

To build `eddl`, clone or download this repository and then, from within the repository, run:

```bash
mkdir build
cd build
cmake ..
make
```

There are some interesting flags to choose what to compile:

- `-DBUILD_PYTHON=ON`: Compiles Python binding
- `-DBUILD_TESTS=ON`: Compiles tests
- `-DBUILD_EXAMPLES=ON`: Compiles examples

> By default, all of them are enabled.

# Getting started

```c++
#include "eddl.h"

int main(int argc, char **argv)
{

  // Download dataset
  eddl.download_mnist();

  // Settings
  int epochs=5;
  int batch_size=1000;
  int num_classes=10;

  // Define network
  layer in=eddl.Input({batch_size, 784});
  layer l = in;  // aux var
  l=eddl.Activation(eddl.Dense(l, 1024), "relu");
  l=eddl.Activation(eddl.Dense(l, 1024), "relu");
  l=eddl.Activation(eddl.Dense(l, 1024), "relu");
  layer out=eddl.Activation(eddl.Dense(l, num_classes),"softmax");
  model net=eddl.Model({in}, {out});

  // View model
  eddl.summary(net);
  eddl.plot(net,"model.pdf");

  // Build model
  eddl.build(net,
            eddl.SGD(0.01,0.9), // Optimizer
            {eddl.LossFunc("soft_cross_entropy")}, // Losses
            {eddl.MetricFunc("categorical_accuracy")}, // Metrics
            eddl.CS_CPU(4) // CPU with 4 threads
            );

  // Load dataset
  tensor x_train = eddl.T("trX.bin");
  tensor y_train = eddl.T("trY.bin");
  tensor x_test = eddl.T("tsX.bin");
  tensor y_test = eddl.T("tsY.bin");

  // Preprocessing
  eddl.div(x_train, 255.0);
  eddl.div(x_test, 255.0);

  // Train model
  eddl.fit(net, {x_train}, {y_train}, batch_size, 5);

  // Evaluate test
  std::cout << "Evaluate train:" << std::endl;
  eddl.evaluate(net, {x_test}, {y_test});
}

```


You can find more examples in the _examples_ folder.

# Tests

To execute all unit tests, go to your build folder and run the following command:

```bash
make test
```


# Continuous build status

| **Build Type**  | **Status** |
|-------------|--------|
| **Linux CPU**   |  [![Build Status](https://travis-ci.org/salvacarrion/EDDL.svg?branch=master)](https://travis-ci.org/salvacarrion/EDDL)|
| **Linux GPU**   |  [![Build Status](https://travis-ci.org/salvacarrion/EDDL.svg?branch=master)](https://travis-ci.org/salvacarrion/EDDL)|
| **Mac OS**      |  [![Build Status](https://travis-ci.org/salvacarrion/EDDL.svg?branch=master)](https://travis-ci.org/salvacarrion/EDDL)|
| **Windows CPU** |  [![Build Status](https://travis-ci.org/salvacarrion/EDDL.svg?branch=master)](https://travis-ci.org/salvacarrion/EDDL)|
| **Windows GPU** |  [![Build Status](https://travis-ci.org/salvacarrion/EDDL.svg?branch=master)](https://travis-ci.org/salvacarrion/EDDL)|


# Python wrapper

If you are not a big fan of C++, you can always try our [PyEDDL](https://github.com/deephealthproject/pyeddl), a python wrapper for this library.
