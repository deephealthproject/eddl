Normalization
=============

BatchNormalization
------------------

.. doxygenfunction:: BatchNormalization

Example:

.. code-block:: c++
   :linenos:

   ...
   l=Dense(l, 1024);
   l=BatchNormalization(l);
   ...


LayerNormalization
------------------

.. doxygenfunction:: LayerNormalization

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

