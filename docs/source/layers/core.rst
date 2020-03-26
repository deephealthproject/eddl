Core
========


Dense
--------

.. doxygenfunction:: Dense

Example:

.. code-block:: c++
   :linenos:

   layer l = Input({784});
   l = Dense(l, 1024);
   l = Dense(l, 1024);
   ...


Embedding
-----------

.. doxygenfunction:: Embedding

Example:

.. code-block:: c++
   :linenos:

   ...

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#core-layers


Reshape
--------

.. doxygenfunction:: Reshape

Example:

.. code-block:: c++
   :linenos:

   // Download mnist
   download_mnist();

   layer l = Input({784});

   // Data augmentation assumes 3D tensors... images:
   l = Reshape(l,{1,28,28});
   // Data augmentation
   ...

   // Come back to 1D tensor for fully connected:
   l = Reshape(l,{-1});

   ...


Flatten
--------

Transforms the input tensor into a 1D-tensor. Alias for ``Reshape(l, {-1})``.

.. doxygenfunction:: Flatten

Example:

.. code-block:: c++
   :linenos:

   // Download mnist
   download_mnist();

   layer l = Input({784});

   // Data augmentation assumes 3D tensors... images:
   l = Reshape(l,{1,28,28});
   // Data augmentation
   ...

   // Come back to 1D tensor for fully connected:
   l = Flatten(l);

   ...


Input
--------

.. doxygenfunction:: Input

Example:

.. code-block:: c++
   :linenos:

   download_mnist();
   layer in = Input({784});



Dropout
--------

.. doxygenfunction:: Dropout

Example:

.. code-block:: c++
   :linenos:

   ...


Select
---------------

Selects a subset of the output tensor using indices (similar to Numpy; the batch is ignored)

.. doxygenfunction:: eddl::Select


Example:

.. code-block:: c++
   :linenos:

   ...
   l = Select(l, {"-1", "20:100", "50:-10", ":"});



Permute
---------------

Permute the axes of the output tensor (the batch is ignored)

.. doxygenfunction:: eddl::Permute


Example:

.. code-block:: c++
   :linenos:

   ...
   l = Permute(l, {0, 2, 1});


Transpose
----------

Permute the last two axes of the output tensor. Alias for ``Permute(l, {0, 2, 1})``.

.. doxygenfunction:: Transpose

Example:

.. code-block:: c++
   :linenos:

   ...
   l = Transpose(l);
