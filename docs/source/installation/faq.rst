FAQ
===


Segmentation fault (core dumped)
--------------------------------

- **CPU:** This is probably because your processor does not support AVX instructions.
- **GPU:** Make sure you are using the computing service: `CS_GPU`.


Protobuf problems
-----------------

If the functions that make use protobuf of such the import/export of ONNX doesn't work:

1) Make sure you have `protobuf` and `libprotobuf` installed
2) In the source code, go to `src/serialization/onnx/` and delete these files: `onnx.pb.cc` and `onnx.pb.cc`.
3) Rebuild them using `protoc --cpp_out=. onnx.proto` (you need to be at `src/serialization/onnx/`)
4) Make use of the cmake flag: `BUILD_PROTOBUF=ON`

.. note::
   We recommend to use the anaconda environment


OpenMP
-------
If you have problems on MacOS, use the ``clang`` by running or addition the following commands to ``.zprofile``

.. code:: bash

    export CC=/usr/local/opt/llvm/bin/clang
    export CXX=/usr/local/opt/llvm/bin/clang++
    export LDFLAGS="-L/usr/local/opt/llvm/lib"
    export CPPFLAGS="-I/usr/local/opt/llvm/include"