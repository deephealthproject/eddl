Core
========


Dense
--------

.. doxygenfunction:: Dense

Example:

.. code-block:: c++

   l = Dense(l, 1024);
   


Embedding
-----------

.. doxygenfunction:: Embedding

Example:

.. code-block:: c++

   l = Embedding(l, 2000, 1, 32);



Reshape
--------

.. doxygenfunction:: Reshape

Example:

.. code-block:: c++

   l = Reshape(l,{1,28,28});

   // Reshape to 1D tensor:
   l = Reshape(l,{-1});



Flatten
--------

Transforms the input tensor into a 1D-tensor. Alias for ``Reshape(l, {-1})``.

.. doxygenfunction:: Flatten

Example:

.. code-block:: c++

   l = Flatten(l);



Input
--------

.. doxygenfunction:: Input

Example:

.. code-block:: c++

   layer in = Input({784});



Dropout
--------

.. doxygenfunction:: Dropout

Example:

.. code-block:: c++

   l = Dropout(l, 0.3);


Select
---------------

Selects a subset of the output tensor using indices (similar to Numpy; the batch is ignored)

.. doxygenfunction:: eddl::Select


Example:

.. code-block:: c++

   l = Select(l, {"-1", "20:100", "50:-10", ":"});



Permute
---------------

Permute the axes of the output tensor (the batch is ignored)

.. doxygenfunction:: eddl::Permute


Example:

.. code-block:: c++

   l = Permute(l, {0, 2, 1});


Transpose
----------

Permute the last two axes of the output tensor. Alias for ``Permute(l, {0, 2, 1})``.

.. doxygenfunction:: Transpose

Example:

.. code-block:: c++

   l = Transpose(l);
