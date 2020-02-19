.. _build-configuration:

Build and configuration
=======================

Configuration
-------------

External dependencies
---------------------

Build and optimization
----------------------




### Building flags

#### Backend support

**CPU support:**
By default the EDDL is build for CPU. If you have any problem and want to compile for CPU, try adding `BUILD_TARGET=CPU` to your cmake options.

.. code: bash
    -DBUILD_TARGET=CPU


**GPU (CUDA) support:**
If you have CUDA installed, you can build EDDL with GPU support by adding `BUILD_TARGET=GPU` to your cmake options.

.. code: bash

    -DBUILD_TARGET=GPU


**FPGA support:**
If available, you can build EDDL with FPGA support by adding `BUILD_TARGET=FPGA` to your cmake options.

.. code: bash

    -DBUILD_TARGET=FPGA


.. note:

    Not yet implemented


#### Additional flags

These flags can enable additional features of the EDDL or help you troubleshooting the installation.

**C++ compiler::**
If you have problems with the default g++ compiler, try setting `EIGEN3_INCLUDE_DIR`, such as:

.. code: bash

    -DCMAKE_CXX_COMPILER=/path/to/c++compiler


**Eigen3:**
At the core of many numerical operations, we use [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page).
If CMake is unable to find Eigen3 automatically, try setting `Eigen3_DIR`, such as:

.. code: bash

    -DEigen3_DIR=/path/to/eigen


**Intel MKL:**
EDDL can leverage Intel's MKL library to speed up computation on the CPU.

To use MKL, include the following cmake option:

.. code: bash

    -DMKL=TRUE


If CMake is unable to find MKL automatically, try setting MKL_ROOT, such as:

.. code: bash

    -DMKL_ROOT="/path/to/MKL"


**CUDA:**
If CMake is unable to find CUDA automatically, try setting `CUDA_TOOLKIT_ROOT_DIR`, such as:

.. code: bash

    -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda


**Build examples:**
To compile the examples, use the setting `BUILD_EXAMPLES`, such as:

.. code: bash

    -DBUILD_EXAMPLES=ON


.. note:

    The examples can be found in `build/targets/`


**Build tests:**
To compile the tests, use the setting `BUILD_TESTS`, such as:

.. code: bash

    -DBUILD_TESTS=ON


**Build shared library:**
To compile the EDDL as a shared library, use the setting `BUILD_SHARED_LIB`, such as:

.. code: bash

    -DBUILD_SHARED_LIB=ON

