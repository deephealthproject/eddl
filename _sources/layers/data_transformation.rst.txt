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

Example:

.. code-block:: c++
   :linenos:

    layer Affine(layer parent, float angle=0, float translate=0, float scale=0, float shear=0, string name="");


Crop
----

.. doxygenfunction:: Crop

Example:

.. code-block:: c++
   :linenos:

    layer Crop(layer parent, vector<int> from_coords, vector<int> to_coords, bool reshape=true, float constant=0.0f, string name="");


CenteredCrop
---------------

.. doxygenfunction:: CenteredCrop

Example:

.. code-block:: c++
   :linenos:

    layer CenteredCrop(layer parent, vector<int> size, bool reshape=true, float constant=0.0f, string name="");


ColorJitter
---------------

.. doxygenfunction:: ColorJitter

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#data-transformations

Example:

.. code-block:: c++
   :linenos:

    layer ColorJitter(layer parent, float brightness=0, float contrast=0, float saturation=0, float hue=0, string name="");  // TODO: Implement


CropScale
---------------

.. doxygenfunction:: CropScale

Example:

.. code-block:: c++
   :linenos:

    layer CropScale(layer parent, vector<int> from_coords, vector<int> to_coords, string da_mode="nearest", float constant=0.0f, string name="");


Cutout
-------

.. doxygenfunction:: Cutout

Example:

.. code-block:: c++
   :linenos:

    layer Cutout(layer parent, vector<int> from_coords, vector<int> to_coords, float constant=0.0f, string name="");



Flip
-------

.. doxygenfunction:: Flip

Example:

.. code-block:: c++
   :linenos:

    layer Flip(layer parent, int axis=0, string name="");

Grayscale
---------

.. doxygenfunction:: Grayscale

Example:

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#data-transformations
    
.. code-block:: c++
   :linenos:

    layer Grayscale(layer parent,  string name="");


HorizontalFlip
---------------------

.. doxygenfunction:: HorizontalFlip

Example:

.. code-block:: c++
   :linenos:

    layer HorizontalFlip(layer parent, string name="");


Pad
--------------

.. doxygenfunction:: Pad

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#data-transformations

Example:

.. code-block:: c++
   :linenos:

    layer Pad(layer parent, vector<int> padding, float constant=0.0f, string name="");



Rotate
-------

.. doxygenfunction:: Rotate

Example:

.. code-block:: c++
   :linenos:

    layer Rotate(layer parent, float angle, vector<int> offset_center={0, 0}, string da_mode="original", float constant=0.0f, string name="");



Scale
-------

.. doxygenfunction:: Scale

Example:

.. code-block:: c++
   :linenos:

    layer Scale(layer parent, vector<int> new_shape, bool reshape=true, string da_mode="nearest", float constant=0.0f, string name="");


Shift
-----------

.. doxygenfunction:: Shift

Example:

.. code-block:: c++
   :linenos:

    layer Shift(layer parent, vector<int> shift, string da_mode="nearest", float constant=0.0f, string name="");


VerticalFlip
---------------------

.. doxygenfunction:: VerticalFlip

Example:

.. code-block:: c++
   :linenos:

    layer VerticalFlip(layer parent, string name="");


Normalize
---------

.. doxygenfunction:: Normalize

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#data-transformations

Example:

.. code-block:: c++
   :linenos:

    layer Normalize(layer parent, string name="");
