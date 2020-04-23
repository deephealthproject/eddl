Data transformation
===================

Deterministic transformations.

.. note::

    **Work in progress**. Not all transformation modes are implemented.

    Currently implemented:

    - ``constant``: The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
    - ``original`` (for rotation): The input is extended by filling all values beyond the edge with the original values


Affine
-------

.. doxygenfunction:: Affine

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#data-transformations



Crop
----

.. doxygenfunction:: Crop

Example:

.. code-block:: c++
   :linenos:

   // Define network
   layer in = Input({784});
   layer l = in;  // Aux var
   l=Reshape(l,{1,28,28});
   // Data transformation
   l = Crop(l, {4,4},{24,24});
   ...

CenteredCrop
---------------

.. doxygenfunction:: CenteredCrop

Example:

.. code-block:: c++
   :linenos:

   // Define network
   layer in = Input({784});
   layer l = in;  // Aux var
   l=Reshape(l,{1,28,28});
   // Data transformation
   l = CenteredCrop(l, {24,24});
   ...

ColorJitter
---------------

.. doxygenfunction:: ColorJitter

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#data-transformations



CropScale
---------------

.. doxygenfunction:: CropScale

Example:

.. code-block:: c++
   :linenos:

   // Define network
   layer in = Input({784});
   layer l = in;  // Aux var
   l=Reshape(l,{1,28,28});
   // Data transformation
   l = CropScale(l, {8,8},{20,20});
   ...

Cutout
-------

.. doxygenfunction:: Cutout

Example:

.. code-block:: c++
   :linenos:

   // Define network
   layer in = Input({784});
   layer l = in;  // Aux var
   l=Reshape(l,{1,28,28});
   // Data transformation
   l = Cutout(l, {0,0},{5,5});
   ...


Flip
-------

.. doxygenfunction:: Flip

Example:

.. code-block:: c++
   :linenos:

   // Define network
   layer in = Input({784});
   layer l = in;  // Aux var
   l=Reshape(l,{1,28,28});
   // Data transformation
   l = Flip(l, 1);
   ...


Grayscale
---------

.. doxygenfunction:: Grayscale


.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#data-transformations
 


HorizontalFlip
---------------------

.. doxygenfunction:: HorizontalFlip

Example:

.. code-block:: c++
   :linenos:

   // Define network
   layer in = Input({784});
   layer l = in;  // Aux var
   l=Reshape(l,{1,28,28});
   // Data transformation
   l = HorizontalFlip(l);
   ...


Pad
--------------

.. doxygenfunction:: Pad

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#data-transformations




Rotate
-------

.. doxygenfunction:: Rotate

Example:

.. code-block:: c++
   :linenos:

   // Define network
   layer in = Input({784});
   layer l = in;  // Aux var
   l=Reshape(l,{1,28,28});
   // Data transformation
   l = Rotate(l, 30.0);
   ...


Scale
-------

.. doxygenfunction:: Scale

Example:

.. code-block:: c++
   :linenos:

   // Define network
   layer in = Input({784});
   layer l = in;  // Aux var
   l=Reshape(l,{1,28,28});
   // Data transformation
   l = Scale(l, {35,35}, false);
   ...

Shift
-----------

.. doxygenfunction:: Shift



VerticalFlip
---------------------

.. doxygenfunction:: VerticalFlip

Example:

.. code-block:: c++
   :linenos:

   // Define network
   layer in = Input({784});
   layer l = in;  // Aux var
   l=Reshape(l,{1,28,28});
   // Data transformation
   l = VerticalFlip(l);
   ...

Normalize
---------

.. doxygenfunction:: Normalize

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#data-transformations

