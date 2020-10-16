Input/Output Operations
========================


Input
-----------------------

load
^^^^^^^^^^^

.. doxygenfunction:: Tensor::load(const string &filename, string format = "")

.. code-block:: c++

    static Tensor* load(const string& filename, string format="");
    template<typename T> static Tensor* load(const string& filename, string format="");
    


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

    t1->save("save.bin");
    t1->save("save.txt");
    t1->save("save.csv");


.. code-block:: c++

    // Create matrix
    Tensor* t1 = Tensor::range(1, 3*100*100);

    // Reshape to a 4D tensor and normalize to RGB (0-255)
    t1->reshape_({1, 3, 100, 100});
    t1->normalize_(0, 255);

    t1->save("save.jpg");

.. image:: ../_static/images/demos/save.jpg
    :width: 256
    :align: center
    :alt: Save tensor as an image

