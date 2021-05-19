Data transformation
===================

Deterministic transformations.



Crop
----

.. doxygenfunction:: Crop

Example:

.. code-block:: c++

   l = Crop(l, {4,4},{24,24});
   


CenteredCrop
---------------

.. doxygenfunction:: CenteredCrop

Example:

.. code-block:: c++

   l = CenteredCrop(l, {24,24});
   


CropScale
---------------

.. doxygenfunction:: CropScale

Example:

.. code-block:: c++

   l = CropScale(l, {8,8},{20,20});
   



Cutout
-------

.. doxygenfunction:: Cutout

Example:

.. code-block:: c++

   l = Cutout(l, {0,0},{5,5});
   



Flip
-------

.. doxygenfunction:: Flip

Example:

.. code-block:: c++

   l = Flip(l, 1);
   



HorizontalFlip
---------------------

.. doxygenfunction:: HorizontalFlip

Example:

.. code-block:: c++

   l = HorizontalFlip(l);
   



Pad
--------------

.. doxygenfunction:: Pad

.. code-block:: c++

   l = Pad(l, {50, 50});



Rotate
-------

.. doxygenfunction:: Rotate

Example:

.. code-block:: c++

   l = Rotate(l, 30.0);
   


Scale
-------

.. doxygenfunction:: Scale

Example:

.. code-block:: c++

   l = Scale(l, {35,35}, false);
   


Shift
-----------

.. doxygenfunction:: Shift



VerticalFlip
---------------------

.. doxygenfunction:: VerticalFlip

Example:

.. code-block:: c++

   l = VerticalFlip(l);


