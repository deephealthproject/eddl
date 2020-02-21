Normalization
=============

BatchNormalization
------------------

Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.

Example:

.. code-block:: c++
   :linenos:

    layer BatchNormalization(layer parent, float momentum = 0.9f, float epsilon = 0.001f, bool affine = true,string name = "");



LayerNormalization
------------------

Applies Layer Normalization over a input.

Example:

.. code-block:: c++
   :linenos:

    layer LayerNormalization(layer parent, float momentum = 0.9f, float epsilon = 0.001f, bool affine = true,string name = "");



GroupNormalization
------------------

Divides the channels into groups and computes within each group the mean and variance for normalization. The computation is independent of batch sizes.

Example:

.. code-block:: c++
   :linenos:

    layer GroupNormalization(layer parent, int groups, float momentum = 0.9f, float epsilon = 0.001f, bool affine = true,string name = "");


