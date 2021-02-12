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

            conda install -c deephealth eddl-cpu

        .. note::

            - Platforms supported: Linux x86/x64 and MacOS

    .. tab:: GPU

        .. code:: bash

            conda install -c deephealth eddl-gpu

        .. note::

            - Platforms supported: Linux x86/x64


.. image:: ../_static/images/logos/homebrew.svg


Using the Homebrew package
--------------------------

A package for EDDL is available on the homebrew package manager.
You need to run both lines, one to add the tap and the other to install the library.

.. code:: bash

    # Add deephealth tap
    brew tap deephealthproject/homebrew-tap

    # Install EDDL
    brew install eddl


.. image:: ../_static/images/logos/cmake.svg


From source with cmake
----------------------

You can also install ``EDDL`` from source with cmake.

.. note::

    **Requirements:**

    * C++ compiler

    * Anaconda

    * [Optional] CUDA Toolkit 10 or later (to compile for GPU)

       * These versions do not support GCC versions later than 8

    * [Optional] cuDNN to speed up training


.. tabs::

    .. tab:: Linux

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
            cmake .. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX

            # [GPU ONLY] CUDA 10/11 does not support gcc versions later than 8.
            # You *might* need to add these flags:
            # -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
            # -DCMAKE_C_COMPILER=$(which gcc-7) \
            # -DCMAKE_CXX_COMPILER=$(which g++-7) \

            make install

    .. tab:: MacOS

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
            cmake .. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX

            make install


See the :doc:`build-options` section for more details about cmake options.

.. note::

    1. You can ignore ``-DCMAKE_PREFIX_PATH`` and ``-DCMAKE_INSTALL_PREFIX`` but it is a good practice to use them
    in order to avoid path conflicts.

    2. To use a specific CUDA version you only need to specify the NVCC location:
    ``-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc``

    3. CUDA 10 and 11 does not support GCC versions later than 8.
    *(Ubuntu 20.04 comes with GCC 9.3.0 by default, so you might need to force a lower version
    with:* ``-DCMAKE_CXX_COMPILER``)

    4. If you want to distribute the resulting shared library, you should use the flag
    ``-DBUILD_SUPERBUILD=ON`` so that we can make specific tunings to our dependencies.

    5. If you don't want to install Anaconda_, you can compile it using:

    .. code:: bash

        cmake .. -DBUILD_SUPERBUILD=ON -DBUILD_TARGET=CUDNN -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CXX_COMPILER=/usr/bin/g++-7


Including EDDL in your project
---------------------------------

The different packages of ``EDDL`` are built with cmake, so whatever the
installation mode you choose, you can add ``EDDL`` to your project using cmake:

.. code:: cmake

    find_package(eddl REQUIRED)
    target_link_libraries(your_target PUBLIC EDDL::eddl)

.. note::

    After ``find_package``, you can access library components with theses variables:
    ``EDDL_ROOT``, ``EDDL_INCLUDE_DIR``, ``EDDL_LIBRARIES_DIR`` and ``EDDL_LIBRARIES``.

.. _Anaconda: https://www.anaconda.com/
