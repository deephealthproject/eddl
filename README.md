<p style="text-align: center;">
  <img src="https://github.com/deephealthproject/eddl/blob/master/docs/sphinx/source/_static/images/logos/logo-eddl-medium.png" alt="EDDL" height="140" width="300">
</p>

-----------------
![build](https://github.com/deephealthproject/eddl/workflows/build/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](https://deephealthproject.github.io/eddl/)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/deephealthproject/eddl)
![GitHub](https://img.shields.io/github/license/deephealthproject/eddl)



**EDDL** is an open source library for Distributed Deep Learning and Tensor Operations in C++ for **CPU**, **GPU** and **FPGA**. EDDL is developed inside the DeepHealth project. For more information about DeepHealth project go to: [deephealth-project.eu](https://deephealth-project.eu/)

## Notice

Please note that EDDL is an ambitious project with few hands on it. Our short-term goal is to have an **stable** version with an easy installation over different platforms. Our second mid-term goal is to provide a good **coverage** of functionalities. And finally, our long-term goal is to improve the **performance** on the different devices.

 *"Plans are only good intentions unless they immediately degenerate into hard work"*

## Key Differences from `main`

This branch makes several changes from `main` branch to address specific challenges and constraints from the [REBECCA-Chip Project](https://www.rebecca-chip.eu/). The key enhancements are:

- **Dependency Installation:** Some dependencies are not bundled with the EDDL installation and must be installed separately.

- **Convolutional Layer Patch:** A patch is included to resolve conflicts with the YOLO family of models when a Convolutional Layer receives an input of an odd size.

- **Model Export:** This branch adds the ability to export a model's complete architecture (topology), not just its weights.


### Exporting a Model

To export a model's topology, include the `import_topology.h` header and use the `Net::describe(const string& fname)` function. This saves the model's architecture to a text file.

```cpp
// Other imports
...


#include "eddl/serialization/topology/import_topology.h"

int main(int argc, char **argv) {


    // Load net
    string model_path = argv[1];
    Net* net = import_net_from_onnx_file(model_path);

    // Build model
    build(net,
          sgd(0.01, 0.9),
          {"soft_cross_entropy"},
          {"categorical_accuracy"},
          CS_CPU(),
          false
    );

    // Save model topology
    net->describe("MODEL_NAME.txt");

    /*
      If the model has constant Tensors, they will be stored with the topology
      as separated bin files.
    */

    // Rest of the code
    ...

  return 0;
}
```

### Importing a Model

You can then import the saved model topology using the `import_net_topology(const string& path)` function. After importing the topology, you must build the model and load the corresponding weights.

```cpp
// Other imports
...


#include "eddl/serialization/topology/import_topology.h"

int main(int argc, char **argv) {


    // Load net
    string model_path = argv[1];
    Net* net = import_net_topology(model_path);

    /*
      If the model need some constant Tensors, that were stored
      with the topology, they need to be in the same "model_path"
      as the topology text file.
    */

    // Build model
    build(net,
          sgd(0.01, 0.9),
          {"soft_cross_entropy"},
          {"categorical_accuracy"},
          CS_CPU(),
          false
    );

    // Load weights
    load(net, "MODEL_NAME_weights.bin");

    // Rest of the code
    ...

  return 0;
}
```



## Installation

Before installing the EDDL, you must first install its two main dependencies: [Eigen](https://gitlab.com/libeigen/eigen) and [Protobuf](https://github.com/protocolbuffers/protobuf).

On a Debian-based system (like Ubuntu), you can install all the necessary build tools with the following commands:

```bash
sudo apt update
sudo apt install git autoconf automake libtool curl make cmake g++ unzip
```

### Eigen

To install the Eigen library, use the following commands:

```bash
git clone https://gitlab.com/libeigen/eigen.git
cd eigen/
git checkout 3.4.0
mkdir build
cd build/
cmake .. -DBUILD_TESTING:BOOL=OFF
sudo make install
```

### Protobuf

To install the Protobuf library, use the following commands:

```bash
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf/
git checkout v3.11.4
git submodule update --init --recursive
./autogen.sh
./configure
make -j$(nproc)
sudo make install
sudo ldconfig
```

### EDDL

Finally, use the following commands to install the EDDL:

```bash
# Check these routes first
export PROTOBUF_INCLUDE_DIRS=/usr/local/include/
export EIGEN_INCLUDE_DIRS=/usr/local/include/eigen3

git clone https://github.com/deephealthproject/eddl.git
cd eddl/
git checkout develop_rebecca
mkdir build
cd build
cmake .. -DBUILD_TARGET=CPU -DBUILD_HPC=OFF -DBUILD_SUPERBUILD=ON -DBUILD_EXAMPLES=OFF -DProtobuf_INCLUDE_DIRS=${PROTOBUF_INCLUDE_DIRS} -DEIGEN3_INCLUDE_DIR=${EIGEN_INCLUDE_DIRS}
make -j$(nproc)
sudo make install
```

## Getting started [here](https://deephealthproject.github.io/eddl/usage/getting_started.html)

## Documentation [here](https://deephealthproject.github.io/eddl/)

## Progress and coverage
- [Deep-learning features](https://github.com/deephealthproject/eddl/blob/master/docs/markdown/eddl_progress.md)
- [Tensor features](https://github.com/deephealthproject/eddl/blob/master/docs/markdown/eddl_progress_tensor.md)

