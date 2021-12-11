Save and load ONNX models
==========================


Exporting to onnx
---------------------

.. doxygenfunction:: save_net_to_onnx_file(Net* net, string path)

Example:

.. code-block:: c++

    save_net_to_onnx_file(net, "my_model.onnx");


Importing onnx files
----------------------

.. doxygenfunction:: import_net_from_onnx_file(string path, int mem = 0, LOG_LEVEL log_level = LOG_LEVEL::INFO)

Example:

.. code-block:: c++

    Net* net = import_net_from_onnx_file("my_model.onnx");


.. doxygenfunction:: import_net_from_onnx_file(string path, vector<int> input_shape, int mem = 0, LOG_LEVEL log_level = LOG_LEVEL::INFO)

Example:

.. code-block:: c++

    Net* net = import_net_from_onnx_file("my_model.onnx", {3, 32, 32});



Simplifying onnx models
----------------------------

Not all onnx models can be loaded by the EDDL since in order to read the model correctly, all its layers and arguments must be supported by our library.

The best way to know if a model is supported by our current version is to try to load it. If import step produces an error,
you try to "standardize" or "simplifying" using *onnx_simplifier_*.

**Installation**

This is only needed is you are not using the conda environment for the EDDL, as we ship ONNX simplifier with it.

.. code-block:: bash

    pip3 install -U pip && pip3 install onnx-simplifier


**Example #1:** Simplifying a "ResNet18" (one input)

.. code-block:: bash

    # Download onnx model from: https://github.com/onnx/models/tree/master/vision/classification/resnet

    # Simplify
    python3 -m onnxsim resnet18-v1-7.onnx resnet18-v1-7_simplify.onnx


**Example #2:** Simplifying a "MobileNet" (input's shape required)

.. code-block:: bash

    # Download onnx model from: https://github.com/onnx/models/tree/master/vision/classification/mobilenet

    # Simplify
    python3 -m onnxsim mobilenetv2-7.onnx mobilenetv2-7_simplified.onnx --input-shape 1,3,224,224


**Example #3:** Simplifying a "TinyYOLOv3" (two inputs, one dynamic)

.. code-block:: bash

    # Download onnx model from: https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov3

    # Simplify
    python3 -m onnxsim tiny-yolov3-11.onnx tiny-yolov3-11_simplified.onnx --dynamic-input-shape --input-shape input_1:1,3,416,416 image_shape:1,2


.. note:

    If the previous steps have not worked for you, you can open the model using Netron_ to check which layers or parameters
    are missing on the EDDL side, and then you can open a new issue requesting these new features.

.. _EDDL_progress: https://github.com/deephealthproject/eddl/blob/master/docs/markdown/eddl_progress.md
.. _onnx_simplifier: https://github.com/daquexian/onnx-simplifier
.. _Netron: https://www.electronjs.org/apps/netron
