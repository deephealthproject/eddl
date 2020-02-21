Data augmentation
=================

This layers performs random transformations over the previous layer.
Ranges are defined using relative coordinates between 0 and 1.

.. note::

    **Work in progress**. Not all transformation modes are implemented.

    Currently implemented:

    - ``constant``: The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
    - ``original`` (for rotation): The input is extended by filling all values beyond the edge with the original values


RandomCrop
----------

Crop the given image at a random location with size `[height, width]`.

Example:

.. code-block:: c++
   :linenos:

   layer RandomCrop(layer parent, vector<int> new_shape, string name= "");


RandomCropScale
---------------

Crop the given image randomly by the size in a range `[a, b]` by and scale it to the parent size.
Example:

.. code-block:: c++
   :linenos:

    layer RandomCropScale(layer parent, vector<float> factor, string da_mode= "nearest", string name= "");


RandomCutout
---------------

Randomly selects a rectangle region in an image and erases its pixels. The random region is defined by the range `[(min_x, max_x), (min_y, max_y)]`, where these are relative values.

Example:

.. code-block:: c++
   :linenos:

    layer RandomCutout(layer parent, vector<float> factor_x, vector<float> factor_y, float constant= 0.0f, string name= "");


RandomFlip
----------

Flip the given image at `axis=n` randomly with a given probability.

Example:

.. code-block:: c++
   :linenos:

   layer RandomFlip(layer parent, int axis, string name= "");


RandomGrayscale
----------------

Converts the given image to grayscale a given probability.

.. note::

    Not yet implemented

Example:

.. code-block:: c++
   :linenos:

       layer RandomGrayscale(layer parent, string name= "");


RandomHorizontalFlip
---------------------

Horizontally flip the given image randomly with a given probability.

Example:

.. code-block:: c++
   :linenos:

   layer RandomHorizontalFlip(layer parent, string name= "");



RandomRotation
--------------

Resize the input image randomly by the size in a range `[a, b]`.

Example:

.. code-block:: c++
   :linenos:

    layer RandomRotation(layer parent, vector<float> factor, vector<int> offset_center= {0, 0}, string da_mode= "original", float constant= 0.0f, string name= "");


RandomScale
--------------

Resize the input image randomly by the size in a range `[a, b]`.

Example:

.. code-block:: c++
   :linenos:

    layer RandomScale(layer parent, vector<float> factor, string da_mode= "nearest", float constant= 0.0f, string name= "");


RandomShift
--------------

Vertically flip the given image randomly with a given probability.


Example:

.. code-block:: c++
   :linenos:

    layer RandomShift(layer parent, vector<float> factor_x, vector<float> factor_y, string da_mode= "nearest", float constant= 0.0f, string name= "");


RandomVerticalFlip
---------------------

Veritically flip the given image randomly with a given probability.

Example:

.. code-block:: c++
   :linenos:

    layer RandomVerticalFlip(layer parent, string name= "");