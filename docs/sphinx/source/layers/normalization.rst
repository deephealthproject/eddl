Normalization
=============

BatchNormalization
------------------

.. doxygenfunction:: eddl::BatchNormalization(layer parent, bool affine, float momentum, float epsilon, string name)

.. doxygenfunction:: eddl::BatchNormalization(layer parent, float momentum = 0.9f, float epsilon = 0.00001f, bool affine = true, string name = "")

Example:

.. code-block:: c++
   :linenos:

   ...
   l=Dense(l, 1024);
   l=BatchNormalization(l);
   ...


LayerNormalization
------------------

.. doxygenfunction:: eddl::LayerNormalization(layer parent, bool affine, float epsilon, string name)
.. doxygenfunction:: eddl::LayerNormalization(layer parent, float epsilon = 0.00001f, bool affine = true, string name = "")

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

