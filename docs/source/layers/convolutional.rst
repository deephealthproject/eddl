Convolutional Layers
=====================

Conv2D
--------

2D Convolution layer.

Example:

.. code-block:: c++
   :linenos:

   layer Conv(layer parent, int filters, const vector<int> &kernel_size,
               const vector<int> &strides = {1, 1}, string padding = "same", int groups = 1,
               const vector<int> &dilation_rate = {1, 1},
               bool use_bias = true, string name = "");


2D Upsampling 
--------------

Upsampling layer.

Example:

.. code-block:: c++
   :linenos:

       layer UpSampling(layer parent, const vector<int> &size, string interpolation = "nearest", string name = "");



Convolutional Transpose
------------------------

Transposed convolution layer.

Example:

.. code-block:: c++
   :linenos:

   layer ConvT(layer parent, int filters, const vector<int> &kernel_size,
                const vector<int> &output_padding, string padding = "same",
                const vector<int> &dilation_rate = {1, 1},
                const vector<int> &strides = {1, 1}, bool use_bias = true, string name = "");