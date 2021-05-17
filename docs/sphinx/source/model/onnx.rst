ONNX
=====


Save to file
-------------

.. doxygenfunction:: save_net_to_onnx_file(Net* net, string path)

Example:

.. code-block:: c++

    save_net_to_onnx_file(net, "my_model.onnx");




Import from file
-----------------

The EDDL supports a subset of the ONNX operators. If you want to check wich layers are supported check out EDDL_progress_.

.. note::

    In case of experiencing an ``error when importing a model created with another library``, it is recommended
    to use onnx_simplifier_ over the onnx file before importing it.

.. doxygenfunction:: import_net_from_onnx_file(string path, int mem = 0, LOG_LEVEL log_level = LOG_LEVEL::INFO)

Example:

.. code-block:: c++

    Net* net = import_net_from_onnx_file("my_model.onnx");


.. doxygenfunction:: import_net_from_onnx_file(string path, vector<int> input_shape, int mem = 0, LOG_LEVEL log_level = LOG_LEVEL::INFO)

Example:

.. code-block:: c++

    Net* net = import_net_from_onnx_file("my_model.onnx", {3, 32, 32});


.. _EDDL_progress: https://github.com/deephealthproject/eddl/blob/master/docs/markdown/eddl_progress.md
.. _onnx_simplifier: https://github.com/daquexian/onnx-simplifier
