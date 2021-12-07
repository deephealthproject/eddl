Convolutions
============


Conv1D
--------

.. doxygenfunction:: eddl::Conv1D

Example:

.. code-block:: c++
    
    l = Conv1D(l, 16, {3}, {1});


Conv2D
--------

.. doxygenfunction:: eddl::Conv2D

Example:

.. code-block:: c++

    l = Conv2D(l, 32, {3,3}, {1,1});


Conv3D
--------

.. doxygenfunction:: eddl::Conv3D

.. code-block:: c++

    l = Conv3D(l, 32, {3, 3, 3}, {1, 1, 1}, "same");


Pointwise Convolution 2D
------------------------

.. doxygenfunction:: eddl::PointwiseConv2D

Example:

.. code-block:: c++

    l = PointwiseConv2D(l, 32, {3,3}, {1,1});


Depthwise Convolution 2D
------------------------

.. doxygenfunction:: eddl::DepthwiseConv2D

Example:

.. code-block:: c++

  l = DepthwiseConv2D(l, {3,3}, {1,1});


2D UpSampling
--------------

Soon to be deprecated. We recommend the use of the Resize layer.

.. doxygenfunction:: eddl::UpSampling2D

Example:

.. code-block:: c++

    l = UpSampling2D(l, {2, 2});


3D UpSampling
--------------

UpSampling for 3D images

.. doxygenfunction:: eddl::UpSampling3D

Example:

.. code-block:: c++

   l = UpSampling3D(l, {32, 32, 32});


2D Convolutional Transpose
----------------------------

.. doxygenfunction:: eddl::ConvT2D

.. code-block:: c++

    l = ConvT2D(l, 32, {3, 3}, {1, 1}, "same");


3D Convolutional Transpose
----------------------------

.. doxygenfunction:: eddl::ConvT3D

.. code-block:: c++

    l = ConvT3D(l, 32, {3, 3, 3}, {1, 1, 1}, "same");
