Core
========

Dense
--------

.. doxygenfunction:: Dense

Example:

.. code-block:: c++
   :linenos:

   layer Dense(layer parent, int ndim, bool use_bias = true,  string name = "");

Embedding
-----------

.. doxygenfunction:: Embedding

Example:

.. code-block:: c++
   :linenos:

    layer Embedding(int input_dim, int output_dim, string name = "");

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#core-layers


Reshape
--------

.. doxygenfunction:: Reshape

Example:

.. code-block:: c++
   :linenos:

    layer Reshape(layer parent, const vector<int> &shape, string name = "");


Transpose
----------

.. doxygenfunction:: Transpose

Example:

.. code-block:: c++
   :linenos:

    layer Transpose(layer parent, string name = "");


Input
--------

.. doxygenfunction:: Input

Example:

.. code-block:: c++
   :linenos:

   layer Input(const vector<int> &shape, string name = "");



Dropout
--------

.. doxygenfunction:: Dropout

Example:

.. code-block:: c++
   :linenos:

   layer Dropout(layer parent, float rate, string name = "");