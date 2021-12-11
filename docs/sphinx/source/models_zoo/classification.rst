Classification
===============

VGG
----

.. doxygenfunction:: eddl::download_vgg16

Example:

.. code-block:: c++

    Net* net = download_vgg16(); // Only convolutional part
    Net* net = download_vgg16(false); // With dense part
    Net* net = download_vgg16(true, {3, 64, 64}); // With new input shape

.. doxygenfunction:: eddl::download_vgg16_bn

Example:

.. code-block:: c++

    Net* net = download_vgg16_bn(); // Only convolutional part
    Net* net = download_vgg16_bn(false); // With dense part
    Net* net = download_vgg16_bn(true, {3, 64, 64}); // With new input shape

.. doxygenfunction:: eddl::download_vgg19

Example:

.. code-block:: c++

    Net* net = download_vgg19(); // Only convolutional part
    Net* net = download_vgg19(false); // With dense part
    Net* net = download_vgg19(true, {3, 64, 64}); // With new input shape

.. doxygenfunction:: eddl::download_vgg19_bn

Example:

.. code-block:: c++

    Net* net = download_vgg19_bn(); // Only convolutional part
    Net* net = download_vgg19_bn(false); // With dense part
    Net* net = download_vgg19_bn(true, {3, 64, 64}); // With new input shape

ResNet
-------

.. doxygenfunction:: eddl::download_resnet18

Example:

.. code-block:: c++

    Net* net = download_resnet18(); // Only convolutional part
    Net* net = download_resnet18(false); // With dense part
    Net* net = download_resnet18(true, {3, 64, 64}); // With new input shape


.. doxygenfunction:: eddl::download_resnet34

Example:

.. code-block:: c++

    Net* net = download_resnet34(); // Only convolutional part
    Net* net = download_resnet34(false); // With dense part
    Net* net = download_resnet34(true, {3, 64, 64}); // With new input shape


.. doxygenfunction:: eddl::download_resnet50

Example:

.. code-block:: c++

    Net* net = download_resnet50(); // Only convolutional part
    Net* net = download_resnet50(false); // With dense part
    Net* net = download_resnet50(true, {3, 64, 64}); // With new input shape


.. doxygenfunction:: eddl::download_resnet101

Example:

.. code-block:: c++

    Net* net = download_resnet101(); // Only convolutional part
    Net* net = download_resnet101(false); // With dense part
    Net* net = download_resnet101(true, {3, 64, 64}); // With new input shape


.. doxygenfunction:: eddl::download_resnet152

Example:

.. code-block:: c++

    Net* net = download_resnet152(); // Only convolutional part
    Net* net = download_resnet152(false); // With dense part
    Net* net = download_resnet152(true, {3, 64, 64}); // With new input shape


DenseNet
---------

.. doxygenfunction:: eddl::download_densenet121

Example:

.. code-block:: c++

    Net* net = download_densenet121(); // Only convolutional part
    Net* net = download_densenet121(false); // With dense part
    Net* net = download_densenet121(true, {3, 64, 64}); // With new input shape
