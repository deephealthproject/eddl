Data transformation
===================

Deterministic transformations.

.. note::

    **Work in progress**. Not all transformation modes are implemented.

    Currently implemented:

    - ``constant``: The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
    - ``original`` (for rotation): The input is extended by filling all values beyond the edge with the original values


Crop
----

Crops the given image at `[(top, left), (bottom, right)]`.

Example:

.. code-block:: c++
   :linenos:

    layer Crop(layer parent, vector<int> from_coords, vector<int> to_coords, bool reshape=true, float constant=0.0f, string name="");


CenteredCrop
---------------

Crops the given image at the center with size (width, height).

Example:

.. code-block:: c++
   :linenos:

    layer CenteredCrop(layer parent, vector<int> size, bool reshape=true, float constant=0.0f, string name="");


ColorJitter
---------------

Randomly change the brightness, contrast and saturation of an image.

.. note::

    Not yet implemented

Example:

.. code-block:: c++
   :linenos:

    layer ColorJitter(layer parent, float brightness=0, float contrast=0, float saturation=0, float hue=0, string name="");  // TODO: Implement


CropScale
---------------

Crop the given image at `[(top, left), (bottom, right)]` and scale it to the parent size.

Example:

.. code-block:: c++
   :linenos:

    layer CropScale(layer parent, vector<int> from_coords, vector<int> to_coords, string da_mode="nearest", float constant=0.0f, string name="");


Cutout
-------

Selects a rectangle region in an image at `[(top, left), (bottom, right)]` and erases its pixels using a constant value.

Example:

.. code-block:: c++
   :linenos:

    layer Cutout(layer parent, vector<int> from_coords, vector<int> to_coords, float constant=0.0f, string name="");



Flip
-------

Flip the given image at `axis=n`.

Example:

.. code-block:: c++
   :linenos:

    layer Flip(layer parent, int axis=0, string name="");

Grayscale
---------

Converts the image to gray scale

Example:

.. note::

    Not yet implemented

.. code-block:: c++
   :linenos:

    layer Grayscale(layer parent,  string name="");


HorizontalFlip
---------------------

Horizontally flip the given image.

Example:

.. code-block:: c++
   :linenos:

    layer HorizontalFlip(layer parent, string name="");


Pad
--------------

Pads the image

.. note::

    Not yet implemented

Example:

.. code-block:: c++
   :linenos:

    layer Pad(layer parent, vector<int> padding, float constant=0.0f, string name="");



Rotate
-------

Rotate the image by angle (degrees)

Example:

.. code-block:: c++
   :linenos:

    layer Rotate(layer parent, float angle, vector<int> offset_center={0, 0}, string da_mode="original", float constant=0.0f, string name="");



Scale
-------

Resize the input image to the given size. `[height, width]`.

Example:

.. code-block:: c++
   :linenos:

    layer Scale(layer parent, vector<int> new_shape, bool reshape=true, string da_mode="nearest", float constant=0.0f, string name="");


Shift
-----------

Shift the input image `[a, b]`.

Example:

.. code-block:: c++
   :linenos:

    layer Shift(layer parent, vector<int> shift, string da_mode="nearest", float constant=0.0f, string name="");


VerticalFlip
---------------------

Vertically flip the given image.

Example:

.. code-block:: c++
   :linenos:

    layer VerticalFlip(layer parent, string name="");


Normalize
---------

Normalize an image with mean and standard deviation.

.. note::

    Not yet implemented

Example:

.. code-block:: c++
   :linenos:

    layer Normalize(layer parent, string name="");
