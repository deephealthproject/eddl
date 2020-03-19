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


Flatten
--------

Transforms the input tensor into a 1D-tensor. Alias for ``Reshape(l, {-1})``.

.. doxygenfunction:: Flatten

Example:

.. code-block:: c++
   :linenos:

    layer Flatten(layer parent, string name = "");


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


Select
---------------

Selects a subset of the output tensor using indices (similar to Numpy; the batch is ignored)

.. doxygenfunction:: eddl::Select


Example:

.. code-block:: c++
   :linenos:

    layer Select(layer l, vector<string> indices, string name="");
    // e.g.: Select(l, {"-1", "20:100", "50:-10", ":"}



Permute
---------------

Permute the axes of the output tensor (the batch is ignored)

.. doxygenfunction:: eddl::Permute


Example:

.. code-block:: c++
   :linenos:

    layer Permute(layer l, vector<int> dims, string name="");
    // e.g.: Permute(l, {0, 2, 1})


Transpose
----------

Permute the last two axes of the output tensor. Alias for ``Permute(l, {0, 2, 1})``.

.. doxygenfunction:: Transpose

Example:

.. code-block:: c++
   :linenos:

    layer Transpose(layer parent, string name = "");
    // e.g.: Transpose(l)
