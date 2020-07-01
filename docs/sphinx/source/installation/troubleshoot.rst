Troubleshoot
============


Segmentation fault (core dumped)
--------------------------------

- **CPU:** This is probably because your processor does not support AVX instructions.
- **GPU:** Make sure you are using the computing service: `CS_GPU`.


Protobuf problems
-----------------

If the ONNX function don't work, it might be due to protobuf. So please, check:

1. Make sure you have ``protobuf`` and ``libprotobuf`` installed in standard paths

2. If you are building the EDDL from source:

    a. Make use of the cmake flag: ``BUILD_PROTOBUF=ON``
    b. Go to ``src/serialization/onnx/`` and delete these files: ``onnx.pb.cc`` and ``onnx.pb.cc``.
    c. Run ``protoc --cpp_out=. onnx.proto`` in the previous directory (``src/serialization/onnx/``) and make sure these files have been generated: ``onnx.pb.cc`` and ``onnx.pb.cc``

.. note::
   Additionally, we recommend to make use of the anaconda environment (see :doc:`installation` section for more details).


OpenMP
-------

If you are using MacOS, we have noticed several problems with default c++ compiler. If one of those problems relate
to ``OpenMP``, we recommend you to use the `clang` compiler. To do so, you can execute the following commands
(or append them to ``.zprofile``):

.. code:: bash

    export CC=/usr/local/opt/llvm/bin/clang
    export CXX=/usr/local/opt/llvm/bin/clang++
    export LDFLAGS="-L/usr/local/opt/llvm/lib"
    export CPPFLAGS="-I/usr/local/opt/llvm/include"

If this doesn't fix your problem, try updating CMake.
As a last resort, you can always disable OpenMP and use the EDDL, by making use of the cmake flag ``-D BUILD_OPENMP=OFF``



(MacOS) Undefined symbols for architecture x86_64
--------------------------------------------------

This error might be due to a conflict with the default compilers. A simple workaround is to force the use ``CClang``
(for instance) for C and C++, and then install the EDDL again:

.. code:: bash

    # Set env variables
    export CC=/usr/local/opt/llvm/bin/clang
    export CXX=/usr/local/opt/llvm/bin/clang++
    export LDFLAGS="-L/usr/local/opt/llvm/lib"
    export CPPFLAGS="-I/usr/local/opt/llvm/include"


Import/Export Numpy files
-------------------------

(Theoretical) Numpy files include a version numbering for the format (independent of the Numpy version).
So if a file it's written using a future format (>= 3.0) that is not backward compatible with the previous importers
and we haven't updated our importer, we won't be able to import the numpy file properly.

If this is your case, please, create a new issue on `github issue`_ and temporally save your numpy file using and older version format (use Numpy).


.. _github issue: https://github.com/deephealthproject/eddl/issues


My model doesn't fit on the GPU but on X deep-learning framework does
---------------------------------------------------------------------

You can change the memory consumption through these memory levels:

- ``full_mem`` (default): No memory bound (highest speed at the expense of the memory requirements)
- ``mid_mem``: Slight memory optimization (good trade-off memory-speed)
- ``low_mem``: Optimized for hardware with restricted memory capabilities.

Take into account that these levels respond to the classical memory-speed trade-off