Image operations
================

Transformations
----------------

Shift
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::shift(vector<int> shift, WrappingMode mode = WrappingMode::Constant, float cval = 0.0f)

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed

    // Shifts t1 50 pixels in y and 100 in x - WP: Constant
    Tensor* t2 = t1->shift({50, 100}, WrappingMode::Constant, 0.0f);
    t2->save("lena_shift.jpg");

    // Other ways
    Tensor::shift(t1, t2, {50, 100}, WrappingMode::Constant, 0.0f);  // static

.. image:: ../_static/images/demos/lena_shift_wm_const.jpg
    :width: 256
    :align: center
    :alt: Shift operation on Lena

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed

    // Shifts t1 50 pixels in y and 100 in x - WP: Original
    Tensor* t2 = t1->shift({50, 100}, WrappingMode::Original, 0.0f);
    t2->save("lena_shift.jpg");

    // Other ways
    Tensor::shift(t1, t2, {50, 100}, WrappingMode::Original, 0.0f);  // static


.. image:: ../_static/images/demos/lena_shift_wm_ori.jpg
    :width: 256
    :align: center
    :alt: Shift operation on Lena (WP: Original)


Rotate
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::rotate(float angle, vector<int> offset_center = {0, 0}, WrappingMode mode = WrappingMode::Constant, float cval = 0.0f)

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed

    // Rotates t1 30 degrees - WP: Constant
    Tensor* t2 = t1->rotate(30.0f, {0,0}, WrappingMode::Constant);
    t2->save("lena_rotate_wm_const.jpg");

    // Other ways
    Tensor::rotate(t1, t2, 30.0f, {0,0}, WrappingMode::Constant);  // Static

.. image:: ../_static/images/demos/lena_rotate_wm_const.jpg
    :width: 256
    :align: center
    :alt: Rotate operation on Lena (WP: Constant)


Scale
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::scale(vector<int> new_shape, WrappingMode mode = WrappingMode::Constant, float cval = 0.0f, bool keep_size = false)

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed

    // Scale to 100x100 pixels
    Tensor* t2 = t1->scale({100, 100}); // keep_size==false
    t2->save("lena_scale_100x100.jpg");

    // Other ways
    Tensor::scale(t1, t2, {100, 100});  // Static

.. image:: ../_static/images/demos/lena_scale_100x100.jpg
    :width: 100
    :align: center
    :alt: Scale operation on Lena (to 100x100)

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed

    // Scale to 880x880 pixels (virtual) but keeping its original size
    Tensor* t2 = t1->scale({880, 880}, WrappingMode::Constant, 0.0f, true); // Keep_size==true
    t2->save("lena_scale_x2_fixed.jpg");

    // Other ways
    Tensor::scale(t1, t2, {880, 880});  // Static

.. image:: ../_static/images/demos/lena_scale_x2_fixed.jpg
    :width: 256
    :align: center
    :alt: Scale operation on Lena (x2, fixed)


Flip
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::flip(int axis = 0)

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed

    // Flip along horizontal axis
    Tensor* t2 = t1->flip(1);
    t2->save("lena_flip_h.jpg");

    // Other ways
    Tensor::flip(t1, t2, 1);  // Static

.. image:: ../_static/images/demos/lena_flip_h.jpg
    :width: 256
    :align: center
    :alt: Flip operation on Lena


Crop
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::crop(vector<int> coords_from, vector<int> coords_to, float cval = 0.0f, bool keep_size = false)

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed

    // Crop a rectangle
    Tensor* t2 = t1->crop({50, 250}, {250, 400});  // keep_size==false
    t2->save("lena_cropped_small.jpg");

    // Other ways
    Tensor::crop(t1, t2, {50, 250}, {250, 400});  // Static

.. image:: ../_static/images/demos/lena_cropped_small.jpg
    :width: 88
    :align: center
    :alt: Crop operation on Lena (Small)


.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed

    // Crop a rectangle
    Tensor* t2 = t1->crop({50, 250}, {250, 400}, 0.0f, true);  // keep_size==true
    t2->save("lena_cropped_big.jpg");

    // Other ways
    Tensor::crop(t1, t2, {50, 250}, {250, 400});  // Static

.. image:: ../_static/images/demos/lena_cropped_big.jpg
    :width: 256
    :align: center
    :alt: Crop operation on Lena (Big)


Crop & Scale
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::crop_scale(vector<int> coords_from, vector<int> coords_to, WrappingMode mode = WrappingMode::Constant, float cval = 0.0f)

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_(); //4D tensor needed

    // Crop and scale
    Tensor* t2 = t1->crop_scale({50, 250}, {250, 400});
    t2->save("lena_crop_scale.jpg");

    // Other ways
    Tensor::crop_scale(t1, t2, {50, 250}, {250, 400});  // Static

.. image:: ../_static/images/demos/lena_crop_scale.jpg
    :width: 256
    :align: center
    :alt: Crop and Scale operation on Lena


Cutout
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cutout(vector<int> coords_from, vector<int> coords_to, float cval = 0.0f)

.. code-block:: c++

    Tensor* t1 = Tensor::load("lena.jpg"); t1->unsqueeze_();  // 4D tensor needed

    // Cutout
    Tensor* t2 = t1->cutout({50, 250}, {250, 400});
    t2->save("lena_cutout.jpg");

    // Other ways
    Tensor::cutout(t1, t2, {50, 250}, {250, 400});  // Static

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