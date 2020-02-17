Data augmentation
=================

RandomFlip
----------

Flip the given image at `axis=n` randomly with a given probability.

Example:

.. code-block::c++
   :linenos:

   layer RandomFlip(layer parent, int axis, string name= "");


RandomRotation
--------------

Rotate the image randomly by an angle defined in a range `[a, b]`.

Example:

.. code-block:: c++
   :linenos:

    layer RandomRotation(layer parent, vector<float> factor, vector<int> offset_center= {0, 0}, string da_mode= "original", float constant= 0.0f, string name= "");


RandomScale
-----------

Resize the input image randomly by the size in a range `[a, b]`.

Example:

.. code-block:: c++
   :linenos:

    layer RandomScale(layer parent, vector<float> factor, string da_mode= "nearest", float constant= 0.0f, string name= "");
