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
  

2D Upsampling 
--------------

.. doxygenfunction:: eddl::UpSampling2D

.. note::

    In future versions this function will call ``scale`` instead of ``repeat``

Example:

.. code-block:: c++

    l = UpSampling2D(l, {2, 2});
    

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