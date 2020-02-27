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


.. doxygenfunction:: RandomCrop

Example:

.. code-block:: c++
   :linenos:

   layer RandomCrop(layer parent, vector<int> new_shape, string name= "");


RandomCropScale
---------------


.. doxygenfunction:: RandomCropScale

Example:

.. code-block:: c++
   :linenos:

    layer RandomCropScale(layer parent, vector<float> factor, string da_mode= "nearest", string name= "");


RandomCutout
---------------


.. doxygenfunction:: RandomCutout

Example:

.. code-block:: c++
   :linenos:

    layer RandomCutout(layer parent, vector<float> factor_x, vector<float> factor_y, float constant= 0.0f, string name= "");


RandomFlip
----------


.. doxygenfunction:: RandomFlip

Example:

.. code-block:: c++
   :linenos:

   layer RandomFlip(layer parent, int axis, string name= "");


RandomGrayscale
----------------

.. doxygenfunction:: RandomGrayscale


.. note::

    Not yet implemented

Example:

.. code-block:: c++
   :linenos:

       layer RandomGrayscale(layer parent, string name= "");


RandomHorizontalFlip
---------------------


.. doxygenfunction:: RandomHorizontalFlip

Example:

.. code-block:: c++
   :linenos:

   layer RandomHorizontalFlip(layer parent, string name= "");



RandomRotation
--------------


.. doxygenfunction:: RandomRotation

Example:

.. code-block:: c++
   :linenos:

    layer RandomRotation(layer parent, vector<float> factor, vector<int> offset_center= {0, 0}, string da_mode= "original", float constant= 0.0f, string name= "");


RandomScale
--------------


.. doxygenfunction:: RandomScale

Example:

.. code-block:: c++
   :linenos:

    layer RandomScale(layer parent, vector<float> factor, string da_mode= "nearest", float constant= 0.0f, string name= "");


RandomShift
--------------


.. doxygenfunction:: RandomShift

Example:

.. code-block:: c++
   :linenos:

    layer RandomShift(layer parent, vector<float> factor_x, vector<float> factor_y, string da_mode= "nearest", float constant= 0.0f, string name= "");


RandomVerticalFlip
---------------------


.. doxygenfunction:: RandomVerticalFlip

Example:

.. code-block:: c++
   :linenos:

    layer RandomVerticalFlip(layer parent, string name= "");