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

A package for EDDL is available on the conda package manager.
You can use one of the following lines according to your needs:

.. code::

    conda install -c deephealth eddl-cpu            # CPU (without ONNX) | Linux, MacOS
    conda install -c deephealth eddl-gpu            # GPU (without ONNX) | Linux
    conda install -c deephealth eddl-gpu-onnx       # GPU (with ONNX)    | Linux


.. image:: ../_static/images/logos/homebrew.svg


Using the Homebrew package
--------------------------

A package for EDDL is available on the homebrew package manager.
You need to run both lines, one to add the tap and the other to install the library.

.. code::

    brew tap deephealthproject/homebrew-tap
    brew install eddl

.. note::

    Only ``CPU`` support.

    If you get an error like ``Undefined symbols for architecture x86_64:``, it might be due to a conflict with
    the default compilers. A simple workaround is to force the use ``CClang`` (for instance) for C and C++,
    and then install the EDDL again:

    .. code::

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

.. code::

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

.. code::

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

.. code::

    find_package(eddl REQUIRED)
    target_link_libraries(your_target PUBLIC eddl)
