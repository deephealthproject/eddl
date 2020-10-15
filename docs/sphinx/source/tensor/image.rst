Image operations
================

Transformations
----------------

Shift
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::shift

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed
    Tensor* t2 = Tensor::empty_like(t1);

    // Shifts t1 50 pixels in y and 100 in x - WP: Constant
    Tensor::shift(t1, t2, {50, 100}, WrappingMode::Constant, 0.0f);
    t2->save("lena_shift.jpg");


.. image:: ../_static/images/demos/lena_shift_wm_const.jpg
    :width: 256
    :align: center
    :alt: Shift operation on Lena

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed
    Tensor* t2 = Tensor::empty_like(t1);

    // Shifts t1 50 pixels in y and 100 in x - WP: Original
    Tensor::shift(t1, t2, {50, 100}, WrappingMode::Original, 0.0f);
    t2->save("lena_shift.jpg");


.. image:: ../_static/images/demos/lena_shift_wm_ori.jpg
    :width: 256
    :align: center
    :alt: Shift operation on Lena (WP: Original)


Rotate
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::rotate

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed
    Tensor* t2 = Tensor::empty_like(t1);

    // Rotates t1 30 degrees - WP: Constant
    Tensor::rotate(t1, t2, 30.0f, {0,0}, WrappingMode::Constant);
    t2->save("lena_rotate_wm_const.jpg");

.. image:: ../_static/images/demos/lena_rotate_wm_const.jpg
    :width: 256
    :align: center
    :alt: Rotate operation on Lena (WP: Constant)


Scale
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::scale

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed
    Tensor* t2 = Tensor::zeros({1, 3, 100, 100});

    // Scale to 100x100 pixels
    Tensor::scale(t1, t2, {100, 100});
    t2->save("lena_scale_100x100.jpg");

.. image:: ../_static/images/demos/lena_scale_100x100.jpg
    :width: 100
    :align: center
    :alt: Scale operation on Lena (to 100x100)

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed
    Tensor* t2 = Tensor::empty_like(t1);

    // Scale to 880x880 pixels (virtual) but keeping its original size
    Tensor::scale(t1, t2, {880, 880});
    t2->save("lena_scale_x2_fixed.jpg");

.. image:: ../_static/images/demos/lena_scale_x2_fixed.jpg
    :width: 256
    :align: center
    :alt: Scale operation on Lena (x2, fixed)


Flip
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::flip(Tensor*, Tensor*, int)

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed
    Tensor* t2 = Tensor::empty_like(t1);

    // Flip along horizontal axis
    Tensor::flip(t1, t2, 1);
    t2->save("lena_flip_h.jpg");

.. image:: ../_static/images/demos/lena_flip_h.jpg
    :width: 256
    :align: center
    :alt: Flip operation on Lena


Crop
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::crop

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed
    Tensor* t2 = Tensor::empty_like(t1);

    //Crop a rectangle
    Tensor::crop(t1, t2, {50, 250}, {250, 400});
    t2->save("lena_cropped_big.jpg");

.. image:: ../_static/images/demos/lena_cropped_big.jpg
    :width: 256
    :align: center
    :alt: Crop operation on Lena (big)


.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed
    Tensor* t2 = Tensor::empty({1, 3, 200, 150});

    //Crop a rectangle
    Tensor::crop(t1, t2, {50, 250}, {250, 400});
    t2->save("lena_cropped_small.jpg");

.. image:: ../_static/images/demos/lena_cropped_small.jpg
    :width: 88
    :align: center
    :alt: Crop operation on Lena (small)


Crop & Scale
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::crop_scale

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed
    Tensor* t2 = Tensor::empty_like(t1);

    //Crop and scale
    Tensor::crop_scale(t1, t2, {50, 250}, {250, 400});
    t2->save("lena_crop_scale.jpg");

.. image:: ../_static/images/demos/lena_crop_scale.jpg
    :width: 256
    :align: center
    :alt: Crop and Scale operation on Lena


Cutout
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cutout

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed
    Tensor* t2 = Tensor::empty_like(t1);

    // Cutout
    Tensor::cutout(t1, t2, {50, 250}, {250, 400});
    t2->save("lena_cutout.jpg");

.. image:: ../_static/images/demos/lena_cutout.jpg
    :width: 256
    :align: center
    :alt: Cutout operation on Lena


Data augmentations
-------------------

Shift Random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::shift_random

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed
    Tensor* t2 = Tensor::empty_like(t1);

    // Shift randomly image +-35% (range for the Y and X axis)
    Tensor::shift_random(t1, t2, {-0.35f, +0.35f}, {-0.35f, +0.35f}, WrappingMode::Constant, 0.0f);
    t2->save("lena_rnd_shift.jpg");

.. image:: ../_static/images/demos/lena_rnd_shift.jpg
    :width: 256
    :align: center
    :alt: Random shift operation on Lena


Rotate Random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::rotate_random

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed
    Tensor* t2 = Tensor::empty_like(t1);

    // Rotate image randomly +-60 degrees, using the coordinates (220, 220) as anchor (from the center)
    Tensor::rotate_random(t1, t2, {-60.0f, +60.0f}, {220, 220});
    t2->save("lena_rnd_rotate.jpg");

.. image:: ../_static/images/demos/lena_rnd_rotate.jpg
    :width: 256
    :align: center
    :alt: Random rotate operation on Lena


Scale Random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::scale_random

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed
    Tensor* t2 = Tensor::empty_like(t1);

    // Scale image randomly +-25%, using NearestNeighbors interpolation
    Tensor::scale_random(t1, t2, {0.75f, 1.25f}, WrappingMode::Nearest);
    t2->save("lena_rnd_scale_nn.jpg");

.. image:: ../_static/images/demos/lena_rnd_scale_nn.jpg
    :width: 256
    :align: center
    :alt: Random scale operation on Lena


Flip Random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::flip_random

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed
    Tensor* t2 = Tensor::empty_like(t1);

    // Flip randomly on the horizontal axis (50% change)
    Tensor::flip_random(t1, t2, 1);
    t2->save("lena_rnd_flip.jpg");

.. image:: ../_static/images/demos/lena_flip_h.jpg
    :width: 256
    :align: center
    :alt: Random flip operation on Lena


Crop Random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::crop_random

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed
    Tensor* t2 = Tensor::empty({1, 3, 256, 256});

    // Crop t1 randomly with a crop size equal to the t2 size
    Tensor::crop_random(t1, t2);
    t2->save("lena_rnd_crop.jpg");

.. image:: ../_static/images/demos/lena_rnd_crop.jpg
    :width: 256
    :align: center
    :alt: Random crop operation on Lena


Crop & Scale Random
^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::crop_scale_random

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed
    Tensor* t2 = Tensor::empty_like(t1);

    // Crop a path with size 65-95% of t1, and scale it to the t2 size
    Tensor::crop_scale_random(t1, t2, {0.65f, 0.95f}, WrappingMode::Nearest);
    t2->save("lena_rnd_crop_scale_nn.jpg");

.. image:: ../_static/images/demos/lena_rnd_crop_scale_nn.jpg
    :width: 256
    :align: center
    :alt: Random Crop & Scale operation on Lena


Cutout Random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cutout_random

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed
    Tensor* t2 = Tensor::empty_like(t1);

    // Cutout a patch with size 10-30% of t1 (height and width)
    Tensor::cutout_random(t1, t2, {0.10f, 0.30f}, {0.10f, 0.30f});
    t2->save("lena_rnd_cutout.jpg");

.. image:: ../_static/images/demos/lena_rnd_cutout.jpg
    :width: 256
    :align: center
    :alt: Random cutout operation on Lena