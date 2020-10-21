Normalization
=============

BatchNormalization
------------------

.. doxygenfunction:: BatchNormalization(layer parent, bool affine, float momentum, float epsilon, string name)

.. doxygenfunction:: BatchNormalization(layer parent, float momentum = 0.99f, float epsilon = 0.001f, bool affine = true, string name = "")

Example:

.. code-block:: c++
   :linenos:

   ...
   l=Dense(l, 1024);
   l=BatchNormalization(l);
   ...


LayerNormalization
------------------

.. doxygenfunction:: LayerNormalization(layer parent, bool affine, float epsilon, string name)
.. doxygenfunction:: LayerNormalization(layer parent, float epsilon = 0.00001f, bool affine = true, string name = "")

Example:

.. code-block:: c++
   :linenos:

   ...
   l=Dense(l, 1024);
   l=LayerNormalization(l);
   ...


GroupNormalization
------------------

.. doxygenfunction:: GroupNormalization

Example:

.. code-block:: c++
   :linenos:

   ...
   l=Dense(l, 1024);
   l=GroupNormalization(l, 8);
   ...

