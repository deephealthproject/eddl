.. _build-configuration:

Build and configuration
=========================

External dependencies
-----------------------

To build EDDL you will need a ``C++11 compiler``

If you want to compile with CUDA support, install:

- NVIDIA CUDA 10 or above
- NVIDIA cuDNN to accelerate primitives (optional)

.. note::

    CUDA 10 and 11 does not support GCC versions later than 8.
    *(Ubuntu 20.04 comes with GCC 9.3.0 by default)*

Also, we highly recommend installing an Anaconda_ environment to manage the external dependencies. You will get controlled dependency versions regardless of your system.

Once you have Anaconda_ installed, you can create and activate our
environment by running the following commands **from the source directory**:

.. code:: bash

    conda env create -f environment-cpu.yml  # -cpu, -gpu, -cudnn
    conda activate eddl

You can also update your environment with:

.. code:: bash

    conda env update -f environment-cpu.yml  # -cpu, -gpu, -cudnn

If you decide to manually install these dependencies in your system (make sure they are at standard paths):

.. code:: yaml

    - cmake>=3.9.2
    - eigen==3.3.7
    - protobuf==3.11.4
    - libprotobuf==3.11.4
    - cudnn==8.0.5.39
    - cudatoolkit-dev==10.1.243
    - gtest
    - graphviz
    - wget
    - doxygen
    - python
    - pip
    - pip:
        - sphinx==3.2.1
        - sphinx_rtd_theme==0.5.0
        - sphinx-tabs==1.3.0
        - breathe==4.22.1

.. note::

    - You can double-check your dependency and versions using this reference `conda list` file: Requirements_
    - When using ``apt-get``, the installed version of the package depends on the distro version (by default). This is important to known, because for instance, on Ubuntu 18.04 ``apt-get install libeigen3-dev`` installs Eigen 3.3.4-4, when the EDDL needs Eigen 3.3.7.


Build and optimization
------------------------

Build
^^^^^^

To build the EDDL, you will need a recent version of cmake. Then, run the following commands from the source directory:

.. code:: bash

    mkdir build/
    cd build/
    cmake ..
    make install

.. note::

    If you are using the conda environment to manage the dependencies, remember to activate it by typing: ``conda activate eddl``


Backend support
^^^^^^^^^^^^^^^^^

You can choose the hardware for which the EDDL will be compiled. By default it is compiled for ``GPU`` but if it is not
not found (or CUDA), the EDDL will automatically fallback to CPU.

- **CPU support:** If you want to compile it for CPU, use the following cmake option:

.. code:: bash

    -DBUILD_TARGET=CPU

.. note::

    Backup option when cuDNN or CUDA is not found


- **GPU (CUDA) support:** If you want to compile it for GPU (CUDA), use the following cmake option:

.. code:: bash

    -DBUILD_TARGET=GPU

.. note::

    Fallback to CPU.
    To use a specific CUDA version you only need to specify the NVCC location: ``-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc``


- **GPU (cuDNN) support:** If you want to compile it for GPU (cuDNN), use the following cmake option:

.. code:: bash

    -DBUILD_TARGET=CUDNN

.. note::

    Enabled by default. If cuDNN is not installed, we will fallback to GPU (CUDA), or to CPU if CUDA is not installed.
    To use a specific CUDA version you only need to specify the NVCC location: ``-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc``


- **FPGA support:** If you want to compile it for FPGA, use the following cmake option:

.. code:: bash

    -DBUILD_TARGET=FPGA



Additional flags
^^^^^^^^^^^^^^^^^

These flags can enable/disable features of the EDDL so that you can optimized and
troubleshoot the compilation process (see: :doc:`troubleshoot`).


- **Prefix path:** Semicolon-separated list of directories specifying installation prefixes to be searched by the ``find_package()``, ``find_program()``, ``find_library()``, ``find_file()``, and ``find_path()`` commands.

.. code:: bash

    -DCMAKE_PREFIX_PATH=/path/to/dir

.. note::

    If using conda, get the path by activating the environment, and typing ``echo $CONDA_PREFIX``


- **Installation paths:** To change the installation paths, use the following cmake option:

.. code:: bash

    -DCMAKE_INSTALL_PREFIX=/path/to/dir

.. note::

    Defaults to ``/usr/local`` on UNIX and ``c:/Program Files`` on Windows.
    If using conda, get the path by activating the environment, and typing ``echo $CONDA_PREFIX``


- **C/C++ compiler:**

.. code:: bash

    -DCMAKE_CXX_COMPILER=/path/to/c++compiler  # /usr/bin/g++-8
    -DCMAKE_C_COMPILER=/path/to/c compiler  # /usr/bin/gcc-8

.. note::

    The default compiler in MacOS has problems with OpenMP. We recommend to install either ``gcc``  or ``clang`` using brew.


- **CUDA compiler:**

.. code:: bash

    -DCMAKE_CUDA_COMPILER=/path/to/cuda compiler  #/usr/bin/nvcc

.. note::

    This flag is needed to known which CUDA Toolkit/cuDNN the user wants to use. By default cmake looks in the ``PATH``.


- **CUDA host compiler:**

.. code:: bash

    -DCMAKE_CUDA_HOST_COMPILER=/path/to/host compiler  # /usr/bin/g++-8

.. note::

    You can also create a symbolic link: (unix) ``sudo ln -s usr/local/cuda-{VERSION} /usr/local/cuda``


- **Eigen3:** At the core of many numerical operations, we use Eigen3_. If CMake is unable to find Eigen3 automatically, try setting ``Eigen3_DIR``, such as:

.. code:: bash

    -DEigen3_DIR=/path/to/eigen  # /usr/lib/cmake/eigen3


- **Use OpenMP:** To enable/disabled OpenMP, use the setting ``BUILD_OPENMP``, such as:

.. code:: bash

    -DBUILD_OPENMP=ON

.. note::

    Enabled by default.
    The default compiler in MacOS has problems with OpenMP. We recommend to install either ``gcc``  or ``clang`` using brew.


- **Use HPC:** To enable/disabled HPC flags, use the setting ``BUILD_HPC``, such as:

.. code:: bash

    -DBUILD_HPC=ON

.. note::

    Enabled by default.
    This enables flags such as: ``-march=native -mtune=native -Ofast -msse -mfpmath=sse -ffast-math -ftree-vectorize``,
    that might cause some units tests to fail due to numerical errors (minor deviations from the value asserted)

- **Use protobuf:** Protobuf allows you to use the ONNX import/export functions, to use them, use the setting ``BUILD_PROTOBUF``, such as:

.. code:: bash

    -DBUILD_PROTOBUF=ON

.. note::

    Enabled by default


- **Build tests:** To compile the tests, use the setting ``BUILD_TESTS``, such as:

.. code:: bash

    -DBUILD_TESTS=ON

.. note::

    Enabled by default.
    The flag ``BUILD_HCP`` needs to be disabled. If not, some tests might not pass due to numerical errors.


- **Build examples:** To compile the examples, use the setting ``BUILD_EXAMPLES``, such as:

.. code:: bash

    -DBUILD_EXAMPLES=ON

.. note::

    Enabled by default


- **Build shared library:** To compile the EDDL as a shared library:

.. code:: bash

    -DBUILD_SHARED_LIBS=ON

.. note::

    Enabled by default

- **Superbuild:** To let the EDDL manage its dependencies automatically:

.. code:: bash

    -DBUILD_SUPERBUILD=ON

.. note::

    Disabled by default. If ``OFF``, cmake will look at your ``CMAKE_PREFIX_PATH``

    If you want to distribute the resulting shared library, you should use the flag
    ``-DBUILD_SUPERBUILD=ON`` so that we can make specific tunings to our dependencies.


.. _Anaconda: https://docs.conda.io/en/latest/miniconda.html
.. _Eigen3: http://eigen.tuxfamily.org/index.php?title=Main_Page
.. _Requirements: https://github.com/deephealthproject/eddl/blob/develop/docs/markdown/bundle/requirements.txt