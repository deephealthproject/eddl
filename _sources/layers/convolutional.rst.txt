Convolutions
============

Conv2D
--------

.. doxygenfunction:: eddl::Conv

Example:

.. code-block:: c++

   layer Conv(layer parent, int filters, const vector<int> &kernel_size,
               const vector<int> &strides = {1, 1}, string padding = "same", int groups = 1,
               const vector<int> &dilation_rate = {1, 1},
               bool use_bias = true, string name = "");


2D Upsampling 
--------------

.. doxygenfunction:: eddl::UpSampling

.. note::

    In future versions this function will call ``scale`` instead of ``repeat``

Example:

.. code-block:: c++
   :linenos:

   layer UpSampling(layer parent, const vector<int> &size, string interpolation = "nearest", string name = "");



Convolutional Transpose
------------------------

.. doxygenfunction:: eddl::UpSampling

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#convolutional-layers

Example:

.. code-block:: c++
   :linenos:

   layer ConvT(layer parent, int filters, const vector<int> &kernel_size,
                const vector<int> &output_padding, string padding = "same",
                const vector<int> &dilation_rate = {1, 1},
                const vector<int> &strides = {1, 1}, bool use_bias = true, string name = "");