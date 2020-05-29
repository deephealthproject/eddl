Normalization
=============

BatchNormalization
------------------

.. doxygenfunction:: eddl::BatchNormalization(layer, bool, float, float, string)

.. doxygenfunction:: eddl::BatchNormalization(layer, float, float, bool, string)

Example:

.. code-block:: c++
   :linenos:

   ...
   l=Dense(l, 1024);
   l=BatchNormalization(l);
   ...


LayerNormalization
------------------

.. doxygenfunction:: eddl::LayerNormalization(layer, bool, float, string)
.. doxygenfunction:: eddl::LayerNormalization(layer, float, bool, string)

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

