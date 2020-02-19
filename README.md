<p style="text-align: center;">
  <img src="https://raw.githubusercontent.com/salvacarrion/salvacarrion.github.io/master/assets/hot-linking/logo-eddl.png" alt="EDDL" height="140" width="300">
</p>

-----------------

**EDDL** is an open source library for Distributd Deep Learning and Tensor Operations in C++. EDDL is developed inside the DeepHealth project.

For more information about DeepHealth project go to: [deephealth-project.eu](https://deephealth-project.eu/)


**Documentation:**
- [Available NN features](https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md)
- [Available Tensor features](https://github.com/deephealthproject/eddl/blob/master/eddl_progress_tensor.md)
- [Doyxigen documentation](http://imagelab.ing.unimore.it/eddl/)



## Installation

### Package managers

> Coming soon...


## Build from source

If you are installing from source, you will need a C++11 compiler. We highly recommend installing an [Anaconda environment](https://docs.conda.io/en/latest/miniconda.html) or also you can use a docker image (see below).

If you want to compile with CUDA support, install

- NVIDIA CUDA 9 or above



### Conda


You will need an [anaconda package manager](https://docs.conda.io/en/latest/miniconda.html)

```bash
git clone https://github.com/deephealthproject/eddl.git
cd eddl
conda env create -f environment.yml
conda activate eddl
mkdir build
cd build
cmake .. -DBUILD_TARGET=CPU  # {CPU, GPU, FPGA}
make -j 4  # num_cores
sudo make install
```

### DOCKER


You will need a [docker engine](https://docs.docker.com/install/)

To build the image, run the following command from the `eddl/` folder:

```
git clone https://github.com/deephealthproject/eddl.git
cd eddl
docker build -t eddl .
```

Then, you can execute it using:

```
docker run -it eddl /bin/bash
```

Or mount it, if you want to **edit the code** in the host machine:

```
docker run -it -v $(pwd):/eddl/ eddl /bin/bash
```

Finally you can compile the code:

```bash
mkdir build
cd build
cmake .. -DBUILD_TARGET=CPU  # {CPU, GPU, FPGA}
make -j 4  # num_cores
sudo make install
```


### **Step by step installation:**

If something fails follow the instructions [here](Installation.md)



## Getting started with eddl

You can find more examples in  `examples/`.

```C++
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "apis/eddl.h"
#include "apis/eddlT.h"

using namespace eddl;

int main(int argc, char **argv) {

    // Download mnist
    download_mnist();

    // Settings
    int epochs = 25;
    int batch_size = 128;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  

    // Convert to 3D for Data Augmentation
    l=Reshape(l,{1,28,28});

    // Data augmentation
    l = RandomCropScale(l, {0.9f, 1.0f});

    // Come back to 1D tensor for fully connected:
    l=Reshape(l,{-1});
    l = ReLu(GaussianNoise(BatchNormalization(Dense(l, 1024)),0.3));
    l = ReLu(GaussianNoise(BatchNormalization(Dense(l, 1024)),0.3));
    l = ReLu(GaussianNoise(BatchNormalization(Dense(l, 1024)),0.3));

    layer out = Activation(Dense(l, num_classes), "softmax");
    model net = Model({in}, {out});

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    // Build model
    build(net,
          sgd(0.01, 0.9), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}) // one GPU
          //CS_CPU(-1) // CPU with maximum threads availables
    );

    // View model
    summary(net);

    setlogfile(net,"mnist_bn_da");

    // Load dataset
    tensor x_train = eddlT::load("trX.bin");
    tensor y_train = eddlT::load("trY.bin");
    tensor x_test = eddlT::load("tsX.bin");
    tensor y_test = eddlT::load("tsY.bin");

    // Preprocessing
    eddlT::div_(x_train, 255.0);
    eddlT::div_(x_test, 255.0);


    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);
    // Evaluate
    printf("Evaluate:\n");
    evaluate(net, {x_test}, {y_test});
}
```

## Python wrapper

If you are not a C++ fan, try [PyEDDL](https://github.com/deephealthproject/pyeddl), a python wrapper for this library.
