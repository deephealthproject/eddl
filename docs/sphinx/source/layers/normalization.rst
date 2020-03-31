Normalization
=============

BatchNormalization
------------------

.. doxygenfunction:: BatchNormalization

Example:

.. code-block:: c++
   :linenos:

    layer BatchNormalization(layer parent, float momentum = 0.9f, float epsilon = 0.001f, bool affine = true,string name = "");



LayerNormalization
------------------

.. doxygenfunction:: LayerNormalization

Example:

.. code-block:: c++
   :linenos:

    layer LayerNormalization(layer parent, float momentum = 0.9f, float epsilon = 0.001f, bool affine = true,string name = "");



GroupNormalization
------------------

.. doxygenfunction:: GroupNormalization

Example:

.. code-block:: c++
   :linenos:

    layer GroupNormalization(layer parent, int groups, float momentum = 0.9f, float epsilon = 0.001f, bool affine = true,string name = "");


