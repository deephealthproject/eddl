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


Squeeze
--------

.. doxygenfunction:: Squeeze

Example:

.. code-block:: c++

   // Squeeze all dimensions (ignoring batch)
   l = Squeeze(l, -1);  // ([B,] 32, 1, 5, 1) => ([B,] 32, 5)

   // Squeeze specific dimension (ignoring batch)
   l = Squeeze(l, 3);  // ([B,] 32, 1, 5, 1) => ([B,] 32, 1, 5)


Unsqueeze
----------

.. doxygenfunction:: Unsqueeze

Example:

.. code-block:: c++

   // Unsqueeze dimension (ignoring batch)
   l = Unsqueeze(l, 0);  // ([B,] 5) => ([B,] 1, 5)

   // Unsqueeze dimension (ignoring batch)
   l = Unsqueeze(l, 1);  // ([B,] 5) => ([B,] 5, 1)


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


Slice
---------------

Alias for Select
Selects a subset of the output tensor using indices (similar to Numpy; the batch is ignored)

.. doxygenfunction:: eddl::Slice


Example:

.. code-block:: c++

   l = Slice(l, {"-1", "20:100", "50:-10", ":"});


Expand
---------------

Returns a layer with singleton dimensions expanded to a larger size.

.. doxygenfunction:: eddl::Expand(layer l, int size, string name="")


Example:

.. code-block:: c++

    // {3, 1, 5, 1, 5} and size=100 => {3, 100, 5, 100, 5}
   l = Expand(l, 100);

Split
---------------


Split a tensor (layer) into a list of tensors (layers). (The batch is ignored).
The indexes mark the split points.

.. doxygenfunction:: eddl::Split


Example:

.. code-block:: c++

    // e.g.: l=> Output shape: {B, 3, 32, 32}
    // vl: {l1, l2, l3}; l1= {B, :1, 32, 32}, l2= {B, 1:2, 32, 32}, l3= {B, 2:, 32, 32}
   vector<layer> vl = Split(l, {1,2}, 0);

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


Resize
-------

Same as Scale but with support for backward operation.

.. doxygenfunction:: Resize

Example:

.. code-block:: c++

   l = Resize(l, {35, 35});


Repeat
-------

.. doxygenfunction:: eddl::Repeat(layer parent, const vector<unsigned int>& repeats, unsigned int axis, string name="");
.. doxygenfunction:: eddl::Repeat(layer parent, unsigned int repeats, unsigned int axis, string name="");

Example:

.. code-block:: c++

    // Example #1:
    l = Repeat(l, 3, 1);  // Repeat depth (axis=3) 3 times
    l = Repeat(l, {3, 2, 1}, 1);  // Repeat col 1 => 3 times; col 2 => 2 times; col 3 => 1 time. (repeat=[3,2,1], axis=1)

    // Example #2:
    l = Reshape(l,{1,28,28});
    l = Repeat(l, 3, 0);  // l => (3, 28, 28)



Tile
-------

.. doxygenfunction:: eddl::Tile(layer parent, const vector<int>& repeats, string name="");

Example:

.. code-block:: c++

    l = Reshape(l,{1,28,28});
    l = Tile(l, {3, 1, 1});  // l => (3, 28, 28)



Broadcasting
-------------

.. doxygenfunction:: eddl::Broadcast(layer parent1, layer parent2, string name="");

Example:

.. code-block:: c++

    // Normalize: (X-mean) / std
    layer mean = ConstOfTensor(new Tensor( {0.485, 0.456, 0.406}, {3}, DEV_CPU));
    layer std = ConstOfTensor(new Tensor( {0.229, 0.224, 0.225}, {3}, DEV_CPU));
    x = Sub(x, Broadcast(mean, x));
    x = Div(x, Broadcast(std, x));
