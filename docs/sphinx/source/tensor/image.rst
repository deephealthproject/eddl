Image operations
================

Transformations
----------------

shift
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::shift

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::shift(t1, t2, {50, 100}, WrappingMode::Constant, 0.0f); // Shifts t1 50 pixels in y and 100 in x.


rotate
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::rotate

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);
    Tensor* t3 = t2->clone();
    Tensor::rotate(t2, t3, 60.0f, {0,0}, WrappingMode::Original); //Rotates t2 60 degrees

scale
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::scale

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = Tensor::zeros({1, 3, 100, 100});
    Tensor::scale(t1, t2, {100, 100}); //Scale the image to 100x100 px



flip
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::flip(Tensor*, Tensor*, int)

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::flip(t1, t2, 0); // Flip along vertical axis

crop
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::crop

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::crop(t1, t2, {0, 250}, {200, 450}); //Crop the rectangle formed by {0,250} and {200,450}

crop_scale
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::crop_scale

.. code-block:: c++


    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::crop_scale(t1, t2, {0, 250}, {200, 450});


cutout
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cutout

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::cutout(t1, t2, {50, 100}, {100, 400});//Fill with zeros the rectangle formed by {50,100} and {100,400}



Data augmentations
-------------------

shift_random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::shift_random

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::shift_random(t1, t2, {0,50}, {10,100}); //Shifts t1 with a random shift value in y between 0 and 50 and in x between 10 and 100.

rotate_random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::rotate_random

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    static void rotate_random(t1, t2, {30,60}); //Rotate t1 with a random rotation factor between 30 and 60 degrees

scale_random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::scale_random

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::scale_random(t1, t2, {10,20}); //Scale t1 with a random scale factor between 10 and 20

flip_random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::flip_random

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::flip_random(t1, t2, 0); //Flip t1 on vertical axis randomly

crop_random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::crop_random

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::crop_random(t1, t2); //Obtain a random crop from t1

crop_scale_random
^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::crop_scale_random

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::crop_scale_random(t1, t2, {10,20}); //Obtain a random crop from t1 and scale it randomly with a factor between 10 and 20

cutout_random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cutout_random

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::cutout_random(t1, t2, {50,60}, {10,100});//Set to 0 pixels in a rectangle defined by a random height between 50 and 60 and a random width between 10 and 100
