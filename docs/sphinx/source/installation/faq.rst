FAQ
===


Is there a Python version?
--------------------------

Yes, the PyEDDL_ is the EDDL version for the Python lovers


Can I contribute?
------------------

Yes, but first open a new issue to explain and discuss your contribution.


Can I control the memory consumption?
-------------------------------------

Yes, we offer several memory levels to control the memory-speed trade-off. These levels are:


- ``full_mem`` (default): No memory bound (highest speed at the expense of the memory requirements)
- ``mid_mem``: Slight memory optimization (good trade-off memory-speed)
- ``low_mem``: Optimized for hardware with restricted memory capabilities.


Is it faster than PyTorch/TensorFlow/etc
----------------------------------------

Check our benchmakrs: eddl_benchmarks_


Is it more memory-efficient than PyTorch/TensorFlow/etc
-------------------------------------------------------

Depends on your memory setting, see the "Can I control the memory consumption?" answer.
Also, you can take a look at our benchmakrs: eddl_benchmarks_


Problems with CUDA and GCC
----------------------------

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

    As of the CUDA 4.1 release, gcc 4.5 is now supported. gcc 4.6 and 4.7 are unsupported.
    As of the CUDA 5.0 release, gcc 4.6 is now supported. gcc 4.7 is unsupported.
    As of the CUDA 6.0 release, gcc 4.7 is now supported.
    As of the CUDA 7.0 release, gcc 4.8 is fully supported, with 4.9 support on Ubuntu 14.04 and Fedora 21.
    As of the CUDA 7.5 release, gcc 4.8 is fully supported, with 4.9 support on Ubuntu 14.04 and Fedora 21.
    As of the CUDA 8 release, gcc 5.3 is fully supported on Ubuntu 16.06 and Fedora 23.
    As of the CUDA 9 release, gcc 6 is fully supported on Ubuntu 16.04, Ubuntu 17.04 and Fedora 25.
    The CUDA 9.2 release adds support for gcc 7
    The CUDA 10.1 release adds support for gcc 8


.. _PyEDDL: https://github.com/deephealthproject/pyeddl
.. _eddl_benchmarks: https://github.com/jofuelo/eddl_benchmark