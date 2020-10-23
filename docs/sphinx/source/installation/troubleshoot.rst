Troubleshoot
============


Compilation
------------


Running an example
^^^^^^^^^^^^^^^^^^^^

If you get an error like: ``Segmentation fault (core dumped)`` run running an example, this could be because you
haven't changed the computing service used in the example. (e.g: It uses de GPU and you don't have one)

Also, if it is using CPU and the library has been compile for CPU, it could be because your processor does not
support AVX instructions.


OpenMP
^^^^^^^^

We have noticed several problems with default c++ compiler in Mac OS. If this is your case, we recommend you to use
the `clang` compiler. To do so, you can execute the following commands (or append them to ``.zprofile``):

.. code:: bash

    export CC=/usr/local/opt/llvm/bin/clang
    export CXX=/usr/local/opt/llvm/bin/clang++
    export LDFLAGS="-L/usr/local/opt/llvm/lib"
    export CPPFLAGS="-I/usr/local/opt/llvm/include"

If this doesn't fix your problem, try updating CMake.
As a last resort, you can always disable OpenMP and use the EDDL, by making use of the cmake flag ``-D BUILD_OPENMP=OFF``



OpenSSL
-------

If you cannot compile the EDDL using the distributed mode due to OpenSSL, you might try these things:

First, make you you have OpenSSL installed:

- Ubuntu/Debian: ``sudo apt-get install libcrypto++-dev libssl-dev``
- MacOS: ``brew install openssl``

If this does not work, check if the following paths are correctly setup:

.. code:: bash

    # NOTE: This is a copy-paste from "brew", but for linux should be quite similar.

    If you need to have openssl@1.1 first in your PATH run:
      echo 'export PATH="/usr/local/opt/openssl@1.1/bin:$PATH"' >> ~/.zshrc

    For compilers to find openssl@1.1 you may need to set:
      export LDFLAGS="-L/usr/local/opt/openssl@1.1/lib"
      export CPPFLAGS="-I/usr/local/opt/openssl@1.1/include"

    For pkg-config to find openssl@1.1 you may need to set:
      export PKG_CONFIG_PATH="/usr/local/opt/openssl@1.1/lib/pkgconfig"


(MacOS) Undefined symbols for architecture x86_64
--------------------------------------------------

First, try to delete the ``build/`` folder and run ``cmake`` again. If this doesn't work, try to force a specific
compiler either with the flag: ``-DCMAKE_CXX_COMPILER`` or by exporting these variables to your environment:

.. code:: bash

    # In MacOS we recommend CLang

    # Set env variables
    export CC=/usr/local/opt/llvm/bin/clang
    export CXX=/usr/local/opt/llvm/bin/clang++
    export LDFLAGS="-L/usr/local/opt/llvm/lib"
    export CPPFLAGS="-I/usr/local/opt/llvm/include"


Memory
------

Memory optimization
^^^^^^^^^^^^^^^^^^^^^^

You can change the memory consumption through three different levels of memory optimization:

- ``full_mem`` (default): No memory bound (highest speed at the expense of the memory requirements)
- ``mid_mem``: Slight memory optimization (good trade-off memory-speed)
- ``low_mem``: Optimized for hardware with restricted memory capabilities.

Take into account that these levels respond to the typical memory-speed trade-off


Protobuf
---------

Missing includes
^^^^^^^^^^^^^^^^^

If you gent an error like this:

.. code:: bash

    .../eddl/src/serialization/onnx/onnx.pb.h:10:10: fatal error: google/protobuf/port_def.inc: No such file or directory
    #include <google/protobuf/port_def.inc>

First, make sure if you have protobuf installed and cmake is detecting the paths correctly:

.. code:: bash

    -- Protobuf include: /usr/include
    -- Protobuf libraries: /usr/lib/x86_64-linux-gnu/libprotobuf.so-lpthread
    -- Protobuf compiler: /usr/bin/protoc

If using conda, first check if you have activated the environment: ``conda activate eddl``.
Then, if the error persists, check if the paths of protobuf outputed by CMake have been mixed up with the paths from
the system (in case protobuf is also installed in the system) like this:

.. code:: bash

    -- Protobuf dir:
    -- Protobuf include: /usr/include
    -- Protobuf libraries: /usr/lib/x86_64-linux-gnu/libprotobuf.so-lpthread
    -- Protobuf compiler: /home/salvacarrion/anaconda3/envs/eddl/bin/protoc

You can try to fix it by forcing cmake to look into the conda env using the flags: ``-DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX`` (We recommend to delete the ``build/`` folder to avoid cache problems)

If the error persists, use the flag `-D BUILD_SUPERBUILD=ON` to download all dependencies and link them automatically to the EDDL.


Missing lib
^^^^^^^^^^^^^^^^^

If you get an error like this...

.. code:: bash

    make[2]: *** No rule to make target 'cmake/third_party/protobuf/lib/libprotobuf.a', needed by 'lib64/libeddl.so'.  Stop.


...it is because when using ``-DBUILD_SUPERBUILD=ON``, all critical dependencies are downloaded and compiled locally. These
compiled libraries can be found in ``eddl/build/cmake/third_party/``. The problem with the protobuf static library is
that in some systems, it can be found either on ``protobuf/lib/`` or ``protobuf/lib64/``.

Because the EDDL looks into ``lib/`` (by default), when the protobuf library appears in ``lib64/`` we cannot find it.
To fix this, create a symbolic link from ``lib64/`` to ``lib/``:

.. code:: bash

    # Inside: eddl/build/cmake/third_party/protobuf/
    ln -s lib64 lib


No matching function
^^^^^^^^^^^^^^^^^^^^^

See question below (``Old version of protoc``).


Old version of protoc
^^^^^^^^^^^^^^^^^^^^^

This is because your version of protobuf is not compatible with the ONNX files we provide (``onnx.pb.h/cc`` and
``onnx.proto``). We know our that the current version of the EDDL v0.7 (at the moment of writing this) works with
protobuf 3.11, to to install it, you can either use the conda environment (recommended):

.. code:: bash

    # Install dependencies
    conda env create -f environment.yml
    conda activate eddl

...or install protobuf manually:

.. code:: bash

    # Variables
    PROTOBUF_VERSION=3.11.4

    # Install requirements
    sudo apt-get install -y wget
    sudo apt-get install -y autoconf automake libtool curl make g++ unzip

    # Download source
    wget https://github.com/protocolbuffers/protobuf/releases/download/v$PROTOBUF_VERSION/protobuf-cpp-$PROTOBUF_VERSION.tar.gz
    tar -xf protobuf-cpp-$PROTOBUF_VERSION.tar.gz

    # Build and install
    cd protobuf-$PROTOBUF_VERSION
    ./configure
    make -j$(nproc)
    make install  # you may need sudo
    ldconfig


If everything is correct, cmake should output something like this, and compile without problems.


.. code::

    -- Use Protobuf: ON
    -- Protobuf dir:
    -- Protobuf include: /usr/local/include
    -- Protobuf libraries: /usr/local/lib/libprotobuf.so-lpthread
    -- Protobuf compiler: /usr/local/bin/protoc


ONNX functions
^^^^^^^^^^^^^^^

If the ONNX functions don't work, it might be due to a problem with protobuf so:

1. Make sure you have ``protobuf`` and ``libprotobuf`` installed in standard paths

2. If you are building the EDDL from source:

    a. Make use of the cmake flag: ``BUILD_PROTOBUF=ON``
    b. Go to ``src/serialization/onnx/`` and delete these files: ``onnx.pb.cc`` and ``onnx.pb.cc``.
    c. Run ``protoc --cpp_out=. onnx.proto`` in the previous directory (``src/serialization/onnx/``) and make sure these files have been generated: ``onnx.pb.cc`` and ``onnx.pb.cc``

.. note::
   Additionally, we recommend to make use of the anaconda environment (see :doc:`installation` section for more details).



CUDA
-----

Unsupported GNU version
^^^^^^^^^^^^^^^^^^^^^^^^

If you gent an error like this:

.. code:: bash

    /usr/include/crt/host_config.h:138:2: error: #error -- unsupported GNU version! gcc versions later than 8 are not supported!
    138 | #error -- unsupported GNU version! gcc versions later than 8 are not supported!

This is because NVIDIA does not support all GNU compilers. Each new version of CUDA support a different range of GNU compilers.
The solution is simply to use a GNU C++ compiler with a version lower or equal to 8.x; you can do this by:

.. code:: bash

    // Exporting these aliases to .bashrc
    export CC=gcc-8
    export CXX=g++-8

    // Or creating a symbolic link to the CUDA GCC
    sudo ln -s /usr/bin/gcc-8 /usr/local/cuda/bin/gcc
    sudo ln -s /usr/bin/g++-8 /usr/local/cuda/bin/g++


Anyway, it is convenient to check which is the maximum GCC version that your CUDA supports.

.. code: bash

    # Answer from SO: https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version#comment56532695_8693381
    # More: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

    As of the CUDA 4.1 release, gcc 4.5 is now supported. gcc 4.6 and 4.7 are unsupported.
    As of the CUDA 5.0 release, gcc 4.6 is now supported. gcc 4.7 is unsupported.
    As of the CUDA 6.0 release, gcc 4.7 is now supported.
    As of the CUDA 7.0 release, gcc 4.8 is fully supported, with 4.9 support on Ubuntu 14.04 and Fedora 21.
    As of the CUDA 7.5 release, gcc 4.8 is fully supported, with 4.9 support on Ubuntu 14.04 and Fedora 21.
    As of the CUDA 8 release, gcc 5.3 is fully supported on Ubuntu 16.06 and Fedora 23.
    As of the CUDA 9 release, gcc 6 is fully supported on Ubuntu 16.04, Ubuntu 17.04 and Fedora 25.
    The CUDA 9.2 release adds support for gcc 7
    The CUDA 10.1 release adds support for gcc 8
    The CUDA 11.1 release adds support for gcc 9

If the problem persists, reinstall CUDA from the `official site <https://developer.nvidia.com/cuda-downloads>`_


IDEs
-----


CLion
^^^^^^

I usually have to set additional flags in order to make CLion able to run the EDDL smoothly:

.. code:: bash

    -DBUILD_TARGET=GPU
    -DCMAKE_C_COMPILER=/usr/bin/gcc-8
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-8
    -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-8
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.1

If you want to run it using the conda environment, add:

.. code:: bash

    -DCMAKE_INSTALL_PREFIX=/path/to/dir
    -DCMAKE_PREFIX_PATH=/path/to/dir

    # Note:
    To get the path, activate the environment a type:
    echo $CONDA_PREFIX
