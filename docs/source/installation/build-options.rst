.. _build-configuration:

Build and configuration
=======================

External dependencies
---------------------

To build EDDL you will need a ``C++11 compiler``

If you want to compile with CUDA support, install:

- NVIDIA CUDA 9 or above
- NVIDIA cuDNN v7 or above

Also, we highly recommend installing an Anaconda_ environment to manage the external dependencies. You will get a high-quality BLAS library (MKL) and controlled dependency versions regardless of your system.

Once you have Anaconda_ installed, you can create and activate our
environment by running the following commands **from the source directory**:

.. code:: bash

    conda env create -f environment.yml
    conda activate eddl

If you decide to manually install the dependencies in your system, make sure they are at standard paths:

.. code::

    graphviz >= 2.38.0
    wget >= 1.19.5
    cmake >= 3.12.2
    openmp >= 6.0.0
    blas >= 1.1
    eigen >= 3.3.7
    cudatoolkit >= 6.0
    zlib >= 1.2.8
    gtest >= 1.8.0
    benchmark >= 1.5.0
    protobuf >= 3.6.1
    libprotobuf >= 3.6.1

    # For development
    git >= 2.19.1
    sphinx >= 1.3.1
    breathe >= 4.9.1
    sphinx_rtd_theme >= 0.1.7



Build and optimization
----------------------

Build
^^^^^

To build the EDDL, you will need a recent version of cmake. Then, run the following commands from the source directory:

.. code::

    mkdir build
    cd build
    cmake ..
    make install

.. note::

    If you are using the conda environment to manage the dependencies, remember to activate it by typing: ``conda activate eddl``


Backend support
^^^^^^^^^^^^^^^

You can choose the hardware for which the EDDL will be compiled. By default it is compile for ``GPU``, and if it is
not found (or CUDA), it is automatically disabled so that it can run of CPU (although a cmake message will be prompted).

- **CPU support:** If you want to compile it for CPU, use the following cmake option:

.. code:: bash

    -DBUILD_TARGET=CPU

.. note::

    Backup option for when there is no GPU, or CUDA is not found.


- **GPU (CUDA) support:** If you have CUDA installed, the EDDL will automatically be compiled for GPU. Additionally, you can force it's use with the following cmake option:

.. code:: bash

    -DBUILD_TARGET=GPU

.. note::

    Default option with fallback to CPU


- **FPGA support:** If available, you can build EDDL with FPGA support using the following cmake option:

.. code:: bash

    -DBUILD_TARGET=FPGA


.. note::

    Not yet implemented


Additional flags
^^^^^^^^^^^^^^^^

These flags can enable/disable features of the EDDL so that you can optimized and
troubleshoot the compilation process (see: :doc:``troubleshoot``).


- **C++ compiler::** If you have problems with the default g++ compiler, try setting ``EIGEN3_INCLUDE_DIR``, such as:

.. code:: bash

    -DCMAKE_CXX_COMPILER=/path/to/c++compiler

.. note::

    On MacOS we recommend to use ``clang`` to avoid problems with OpenMP


- **Eigen3:** At the core of many numerical operations, we use Eigen3_. If CMake is unable to find Eigen3 automatically, try setting ``Eigen3_DIR``, such as:

.. code:: bash

    -DEigen3_DIR=/path/to/eigen


- **Intel MKL:** EDDL can leverage Intel's MKL library to speed up computation on the CPU.

To use MKL, include the following cmake option:

.. code:: bash

    -DMKL=TRUE


If CMake is unable to find MKL automatically, try setting MKL_ROOT, such as:

.. code:: bash

    -DMKL_ROOT="/path/to/MKL"


- **CUDA:** If CMake is unable to find CUDA automatically, try setting ``CUDA_TOOLKIT_ROOT_DIR``, such as:

.. code:: bash

    -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda


- **Use OpenMP:** To enable/disabled OpenMP, use the setting ``BUILD_OPENMP``, such as:

.. code:: bash

    -DBUILD_OPENMP=ON

.. note::

    Enabled by default


- **Use protobuf:** Protobuf allows you to use the ONNX import/export functions, to use them, use the setting ``BUILD_PROTOBUF``, such as:

.. code:: bash

    -DBUILD_PROTOBUF=ON

.. note::

    Disabled by default (this dependency can be tricky to install)

- **Build for High-Performance Computing:** To compile the EDDL using aggressive flags to speed-up the code, use the following cmake option:

.. code:: bash

    -DBUILD_HPC=ON

.. note::

    Disabled by default (Use it carefully, your processor might not support these optimizations)


- **Build tests:** To compile the tests, use the setting ``BUILD_TESTS``, such as:

.. code:: bash

    -DBUILD_TESTS=ON

.. note::

    Enabled by default

- **Build examples:** To compile the examples, use the setting ``BUILD_EXAMPLES``, such as:

.. code:: bash

    -DBUILD_EXAMPLES=ON


.. note::

    Enabled by default


- **Build tests:** To compile the tests, use the setting ``BUILD_TESTS``, such as:

.. code:: bash

    -DBUILD_TESTS=ON

.. note::

    Enabled by default

- **Build shared library:** To compile the EDDL as a shared library, use the setting ``BUILD_SHARED_LIB``, such as:

.. code:: bash

    -DBUILD_SHARED_LIB=ON

.. note::

    Enabled by default (if ``OFF``, it will build a static library)


- **Installation paths:** To change the installation paths, use the following cmake option:

.. code:: bash

    -DCMAKE_INSTALL_PREFIX=/path/to/dir

.. note::

    Defaults to ``/usr/local`` on UNIX and ``c:/Program Files`` on Windows.


.. _Anaconda: https://docs.conda.io/en/latest/miniconda.html
.. _Eigen3: http://eigen.tuxfamily.org/index.php?title=Main_Page
