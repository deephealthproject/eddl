ONNX
=====


Save to file
-------------

.. doxygenfunction:: save_net_to_onnx_file( Net *net, string path )

Example:

.. code-block:: c++

    save_net_to_onnx_file(net, "my_model.onnx");




Import from file
-----------------

.. doxygenfunction:: import_net_from_onnx_file(std::string path, int mem = 0)

Example:

.. code-block:: c++

    Net* net = import_net_from_onnx_file("my_model.onnx", DEV_CPU);