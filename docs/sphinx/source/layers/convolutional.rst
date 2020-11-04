Convolutions
============


Conv1D
--------

.. doxygenfunction:: eddl::Conv1D

Example:

.. code-block:: c++
    
    l = Conv1D(l,16, {3},{1});


Conv2D
--------

.. doxygenfunction:: eddl::Conv

Example:

.. code-block:: c++

    l = Conv(l,32, {3,3},{1,1});
    
    

2D Upsampling 
--------------

.. doxygenfunction:: eddl::UpSampling

.. note::

    In future versions this function will call ``scale`` instead of ``repeat``

Example:

.. code-block:: c++

    l = UpSampling(l, {2, 2});
    

Convolutional Transpose
------------------------

.. doxygenfunction:: eddl::ConvT

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/docs/markdown/eddl_progress.md#convolutional-layers

