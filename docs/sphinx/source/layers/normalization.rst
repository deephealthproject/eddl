Normalization
=============

BatchNormalization
------------------

.. doxygenfunction:: BatchNormalization(layer parent, bool affine, float momentum, float epsilon, string name)

.. doxygenfunction:: BatchNormalization(layer parent, float momentum = 0.99f, float epsilon = 0.001f, bool affine = true, string name = "")

Example:

.. code-block:: c++

   l = BatchNormalization(l);
   


LayerNormalization
------------------

.. doxygenfunction:: LayerNormalization(layer parent, bool affine, float epsilon, string name)
.. doxygenfunction:: LayerNormalization(layer parent, float epsilon = 0.00001f, bool affine = true, string name = "")

Example:

.. code-block:: c++

   l = LayerNormalization(l);
   


GroupNormalization
------------------

.. doxygenfunction:: GroupNormalization

Example:

.. code-block:: c++

   l = GroupNormalization(l, 8);
   

