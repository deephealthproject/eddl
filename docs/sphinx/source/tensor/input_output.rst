Input/Output Operations
========================

.. note::
   A practical example of the vast majority of the operations in this page are included in a working example in our `GitHub respository <https://github.com/deephealthproject/eddl/blob/master/examples/tensor/eddl_io.cpp>`_

Input
-----------------------

load
^^^^^^^^^^^

.. doxygenfunction:: Tensor::load(const string &filename, string format = "")

.. code-block:: c++

    Tensor* t1 = Tensor::load("mytensor.bin");
    // [
    // [1.00 2.00 3.00]
    // [4.00 5.00 6.00]
    // [7.00 8.00 9.00]
    // ]


Output
-----------------------

save
^^^^^^^^

.. doxygenfunction:: Tensor::save

.. code-block:: c++

    // Create matrix
    Tensor* t1 = Tensor::range(1, 9); t1->reshape_({3, 3});
    // [
    // [1.00 2.00 3.00]
    // [4.00 5.00 6.00]
    // [7.00 8.00 9.00]
    // ]

    t1->save("mytensor.bin");
    t1->save("mytensor.txt");
    t1->save("mytensor.csv");


.. code-block:: c++

    // Create matrix
    Tensor* t1 = Tensor::range(1, 3*100*100);

    // Reshape to a 4D tensor and normalize to RGB (0-255)
    t1->reshape_({1, 3, 100, 100});
    t1->normalize_(0, 255);

    t1->save("mytensor.jpg");

.. image:: ../_static/images/demos/mytensor.jpg
    :width: 100
    :align: center
    :alt: Save tensor as an image

