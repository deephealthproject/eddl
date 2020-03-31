.. raw:: html

   <style>
   .rst-content .section>img {
       width: 30px;
       margin-bottom: 0;
       margin-top: 0;
       margin-right: 15px;
       margin-left: 15px;
       float: left;
   }
   </style>


Installation
============

.. image:: ../_static/images/logos/conda.svg


Using the Conda package
-----------------------

A package for EDDL is available on Anaconda_.
You can use one of the following lines according to your needs:

.. tabs::

    .. tab:: CPU

        .. code:: bash

            conda activate
            conda install -c deephealth eddl-cpu

        .. note::

            - It does not include ONNX support
            - Platforms supported: Linux x86/x64 and MacOS

    .. tab:: GPU

        .. code:: bash

            conda activate
            conda install -c deephealth eddl-gpu

        .. note::

            - It does not include ONNX support
            - Platforms supported: Linux x86/x64

    .. tab:: GPU-ONNX

        .. code:: bash

            conda activate
            conda install -c deephealth eddl-gpu-onnx

        .. note::

            - Platforms supported: Linux x86/x64


Enabling ONNX features
~~~~~~~~~~~~~~~~~~~~~

If you want to enable the ONNX features, in addition to installing the ``eddl-gpu-onnx`` binary, you need to
install ``protobuf`` manually:


.. code:: bash

    # Download source
    PROTOBUF_VERSION=3.11.4
    sudo apt-get install -y wget
    sudo apt-get install -y autoconf automake libtool curl make g++ unzip
    wget https://github.com/protocolbuffers/protobuf/releases/download/v$PROTOBUF_VERSION/protobuf-cpp-$PROTOBUF_VERSION.tar.gz
    tar -xf protobuf-cpp-$PROTOBUF_VERSION.tar.gz

    # Build and install
    cd protobuf-$PROTOBUF_VERSION
    ./configure
    make -j$(nproc)
    make install
    ldconfig


.. image:: ../_static/images/logos/homebrew.svg


Using the Homebrew package
--------------------------

A package for EDDL is available on the homebrew package manager.
You need to run both lines, one to add the tap and the other to install the library.

.. code:: bash

    # Install Homebrew
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

    # Add deephealth tap
    brew tap deephealthproject/homebrew-tap

    # Install EDDL
    brew install eddl

.. note::

    Only ``CPU`` support.

    If you get an error like ``Undefined symbols for architecture x86_64:``, it might be due to a conflict with
    the default compilers. A simple workaround is to force the use ``CClang`` (for instance) for C and C++,
    and then install the EDDL again:

    .. code:: bash

        # Set env variables
        export CC=/usr/local/opt/llvm/bin/clang
        export CXX=/usr/local/opt/llvm/bin/clang++
        export LDFLAGS="-L/usr/local/opt/llvm/lib"
        export CPPFLAGS="-I/usr/local/opt/llvm/include"

        # Add tap
        brew tap deephealthproject/homebrew-tap

        # Uninstall and install the EDDL
        brew uninstall eddl
        brew install eddl


.. image:: ../_static/images/logos/cmake.svg

From source with cmake
----------------------

You can also install ``EDDL`` from source with cmake. In order to manage the external dependencies we recommend to
install Anaconda (see the :doc:`build-options` section for more details about external dependencies).

On Unix platforms, from the source directory:

.. code:: bash

    # Download source code
    git clone https://github.com/deephealthproject/eddl.git
    cd eddl/

    # Install dependencies
    conda env create -f environment.yml
    conda activate eddl

    # Build and install
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=path_to_prefix ..
    make install

On Windows platforms, from the source directory:

.. code:: bash

    # Download source code
    git clone https://github.com/deephealthproject/eddl.git
    cd eddl/

    # Install dependencies
    conda env create -f environment.yml
    conda activate eddl

    # Build and install
    mkdir build
    cd build
    cmake -G "NMake Makefiles" -DCMAKE_INSTALL_PREFIX=path_to_prefix ..
    nmake
    nmake install

``path_to_prefix`` is the absolute path to the folder where cmake searches for
dependencies and installs libraries. ``EDDL`` installation from cmake assumes
this folder contains ``include`` and ``lib`` subfolders.

See the :doc:`build-options` section for more details about cmake options.

.. note::

    You can ignore the flag ``-DCMAKE_INSTALL_PREFIX`` if you prefer to use the standard paths


Including EDDL in your project
---------------------------------

The different packages of ``EDDL`` are built with cmake, so whatever the
installation mode you choose, you can add ``EDDL`` to your project using cmake:

.. code:: cmake

    find_package(eddl REQUIRED)
    target_link_libraries(your_target PUBLIC eddl)

.. _Anaconda: https://www.anaconda.com/
