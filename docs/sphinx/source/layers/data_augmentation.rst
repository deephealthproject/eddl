Data augmentation
=================

This layers perform random transformations over the previous layer.
Ranges are defined using relative coordinates between 0 and 1.

.. note::

    **Work in progress**. Not all transformation modes are implemented.

    Currently implemented:

    - ``constant``: The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
    - ``original`` (for rotation): The input is extended by filling all values beyond the edge with the original values


RandomAffine
-------------


.. doxygenfunction:: RandomAffine

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#data-augmentations




RandomCrop
----------


.. doxygenfunction:: RandomCrop

Example:

.. code-block:: c++
   :linenos:

   // Define network
   layer in = Input({784});
   layer l = in;  // Aux var
   // Data augmentation assumes 3D tensors... images:
   l=Reshape(l,{1,28,28});
   // Data augmentation
   l = RandomCrop(l, {20, 20});
   ...


RandomCropScale
---------------


.. doxygenfunction:: RandomCropScale

Example:

.. code-block:: c++
   :linenos:

   // Define network
   layer in = Input({784});
   layer l = in;  // Aux var
   // Data augmentation assumes 3D tensors... images:
   l=Reshape(l,{1,28,28});
   // Data augmentation
   l = RandomCropScale(l, {0.9f, 1.0f});
   ...


RandomCutout
---------------


.. doxygenfunction:: RandomCutout

Example:

.. code-block:: c++
   :linenos:

   // Define network
   layer in = Input({784});
   layer l = in;  // Aux var
   // Data augmentation assumes 3D tensors... images:
   l=Reshape(l,{1,28,28});
   // Data augmentation
   // The values of x_min, x_max, y_min and y_max should be between 0.0 and 1.0
   l = RandomCutout(l, {0.3, 0.7},{0.3,0.9});
   ...

RandomFlip
----------


.. doxygenfunction:: RandomFlip

Example:

.. code-block:: c++
   :linenos:

   // Define network
   layer in = Input({784});
   layer l = in;  // Aux var
   // Data augmentation assumes 3D tensors... images:
   l=Reshape(l,{1,28,28});
   // Data augmentation
   l = RandomFlip(l, 0);
   ...


RandomGrayscale
----------------

.. doxygenfunction:: RandomGrayscale


.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#data-augmentations



RandomHorizontalFlip
---------------------


.. doxygenfunction:: RandomHorizontalFlip

Example:

.. code-block:: c++
   :linenos:

   // Define network
   layer in = Input({784});
   layer l = in;  // Aux var
   // Data augmentation assumes 3D tensors... images:
   l=Reshape(l,{1,28,28});
   // Data augmentation
   l = RandomHorizontalFlip(l);
   ...



RandomRotation
--------------


.. doxygenfunction:: RandomRotation

Example:

.. code-block:: c++
   :linenos:

   // Define network
   layer in = Input({784});
   layer l = in;  // Aux var
   // Data augmentation assumes 3D tensors... images:
   l=Reshape(l,{1,28,28});
   // Data augmentation
   l = RandomRotation(l, {-20,30});
   ...

RandomScale
--------------


.. doxygenfunction:: RandomScale

Example:

.. code-block:: c++
   :linenos:

   // Define network
   layer in = Input({784});
   layer l = in;  // Aux var
   // Data augmentation assumes 3D tensors... images:
   l=Reshape(l,{1,28,28});
   // Data augmentation
   l = RandomScale(l, {0.9,1.1});

RandomShift
--------------


.. doxygenfunction:: RandomShift

Example:

.. code-block:: c++
   :linenos:

   // Define network
   layer in = Input({784});
   layer l = in;  // Aux var
   // Data augmentation assumes 3D tensors... images:
   l=Reshape(l,{1,28,28});
   // Data augmentation
   // The shift factors must fall within the range [-1.0, 1.0]
   l = RandomShift(l, {-0.3,0.3},{-0.2, 0.2});

RandomVerticalFlip
---------------------


.. doxygenfunction:: RandomVerticalFlip

Example:

.. code-block:: c++
   :linenos:
   
   // Define network
   layer in = Input({784});
   layer l = in;  // Aux var
   // Data augmentation assumes 3D tensors... images:
   l=Reshape(l,{1,28,28});
   // Data augmentation
   l = RandomVerticalFlip(l);