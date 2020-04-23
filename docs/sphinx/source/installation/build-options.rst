.. _build-configuration:

Build and configuration
=======================

External dependencies
---------------------

To build EDDL you will need a ``C++11 compiler``

If you want to compile with CUDA support, install:

- NVIDIA CUDA 9 or above

Also, we highly recommend installing an Anaconda_ environment to manage the external dependencies. You will get a high-quality BLAS library (MKL) and controlled dependency versions regardless of your system.

Once you have Anaconda_ installed, you can create and activate our
environment by running the following commands **from the source directory**:

.. code:: bash

    conda env create -f environment.yml
    conda activate eddl

If you decide to manually install the dependencies in your system, make sure they are at standard paths:

.. code::

    - cmake>=3.9.2
    - eigen>=3.3.7
    - zlib=1.2.*
    - protobuf=3.11.*
    - cudatoolkit
    - gtest
    - graphviz  # Build & Run
    - wget
    - doxygen  # Docs
    - python
    - pip

    - pip:
        - sphinx
        - breathe
        - sphinx_rtd_theme
        - sphinx-tabs


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


- **Prefix path:** Semicolon-separated list of directories specifying installation prefixes to be searched by the ``find_package()``, ``find_program()``, ``find_library()``, ``find_file()``, and ``find_path()`` commands.

.. code:: bash

    -DCMAKE_PREFIX_PATH=/path/to/directory


- **Installation paths:** To change the installation paths, use the following cmake option:

.. code:: bash

    -DCMAKE_INSTALL_PREFIX=/path/to/dir

.. note::

    Defaults to ``/usr/local`` on UNIX and ``c:/Program Files`` on Windows.


- **C++ compiler:** If you have problems with the default g++ compiler, try setting ``EIGEN3_INCLUDE_DIR``, such as:

.. code:: bash

    -DCMAKE_CXX_COMPILER=/path/to/c++compiler

.. note::

    On MacOS we recommend to use ``clang`` to avoid problems with OpenMP


- **CUDA compiler:** If cmake have problems finding your cuda compiler, try setting ``CMAKE_CUDA_COMPILER``, such as:

.. code:: bash

    -DCMAKE_CUDA_COMPILER=/path/to/cuda compiler

.. note::

    You can also create a symbolic link: (unix) ``sudo ln -s usr/local/cuda-{VERSION} /usr/local/cuda``


- **CUDA Toolkit:** If CMake is unable to find CUDA automatically, try setting ``CUDA_TOOLKIT_ROOT_DIR``, such as:

.. code:: bash

    -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda


- **Eigen3:** At the core of many numerical operations, we use Eigen3_. If CMake is unable to find Eigen3 automatically, try setting ``Eigen3_DIR``, such as:

.. code:: bash

    -DEigen3_DIR=/path/to/eigen


- **Use OpenMP:** To enable/disabled OpenMP, use the setting ``BUILD_OPENMP``, such as:

.. code:: bash

    -DBUILD_OPENMP=ON

.. note::

    Enabled by default


- **Use protobuf:** Protobuf allows you to use the ONNX import/export functions, to use them, use the setting ``BUILD_PROTOBUF``, such as:

.. code:: bash

    -DBUILD_PROTOBUF=ON

.. note::

    Enabled by default


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


- **Build shared library:** To compile the EDDL as a shared library, use the setting ``BUILD_SHARED_LIBS``, such as:

.. code:: bash

    -DBUILD_SHARED_LIBS=ON

.. note::

    Enabled by default

- **Superbuild:** To let the EDDL manage its dependencies automatically, use the setting ``BUILD_SUPERBUILD``:

.. code:: bash

    -DBUILD_SUPERBUILD=ON

.. note::

    Disabled by default. If ``OFF``, cmake will look at your ``CMAKE_PREFIX_PATH``

    If you want to distribute the resulting shared library, you should use the flag
    ``-DBUILD_SUPERBUILD=ON`` so that we can make specific tunings to our dependencies.


.. _Anaconda: https://docs.conda.io/en/latest/miniconda.html
.. _Eigen3: http://eigen.tuxfamily.org/index.php?title=Main_Page
