Indexing & Sorting
===================

Indexing
--------------

Nonzero
^^^^^^^^^

.. doxygenfunction:: Tensor::nonzero

Example:

.. code-block:: c++

   Tensor* t1 = new Tensor({0,10,20,0,50,0}, {2, 3});
   // [
   // [0.00 10.00 20.00]
   // [0.00 50.00 0.00]
   // ]

   Tensor* t2 = t1->nonzero();
   // [1 2 4]


Where
^^^^^^^^^

.. doxygenfunction:: Tensor::where(Tensor *condition, Tensor *A, Tensor *B)


Example:

.. code-block:: c++

   Tensor* t1 = Tensor::randn({2, 3});
   // [
   // [-0.44 -0.22 0.12]
   // [1.37 0.50 -0.72]
   // ]

   Tensor* t2 = Tensor::zeros({2, 3});
   // [
   // [0.00 0.00 0.00]
   // [0.00 0.00 0.00]
   // ]

   Tensor* condition = t1->greater(t2);
   // [
   // [0 0 1]
   // [1 1 0]
   // ]

   Tensor* t3 = Tensor::where(condition, t1, t2);
   // [
   // [0.00 0.00 0.12]
   // [1.37 0.50 0.00]
   // ]


Select
^^^^^^^^^

.. doxygenfunction:: Tensor::select(const vector<string> &indices)

Example:

.. code-block:: c++

   Tensor* t1 = Tensor::randn({3, 4});
   // [
   // [2.30 -1.13 -0.61 0.04]
   // [-0.95 0.96 -0.32 -0.61]
   // [0.29 -0.14 -0.42 -0.04]
   // ]

   Tensor* t2 = t1->select({"0", "2"});  // Number at (0, 2)
   // [
   // [-0.61]
   // ]

   Tensor* t3 = t1->select({":", "2"}); // 3rd column
   // [
   // [-0.61]
   // [-0.32]
   // [-0.42]
   // ]

   Tensor* t4 = t1->select({"0:2", "-2:"}); // 3rd and 4th column from 1st and 2nd row
   // [
   // [-0.61 0.04]
   // [-0.32 -0.61]
   // ]



Set Select
^^^^^^^^^^^

.. doxygenfunction:: set_select(const vector<string> &indices, float value)

Example:

.. code-block:: c++

    Tensor* t1 = Tensor::ones({3, 4});
    t1->set_select({"0", "2"}, 2.0f); // Set element (0, 2) to "2"
    // [
    // [1.00 1.00 2.00 1.00]
    // [1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00]
    // ]

    Tensor* t2 = Tensor::ones({3, 4});
    t2->set_select({":", "2"}, 5.0f);  // Set 2nd column to "5"
    // [
    // [1.00 1.00 5.00 1.00]
    // [1.00 1.00 5.00 1.00]
    // [1.00 1.00 5.00 1.00]
    // ]

    Tensor* t3 = Tensor::ones({3, 4});
    t3->set_select({"0:2", "-2:"}, 7.0f);  // Set 3rd and 4th column from 1st and 2nd row to "7"
    // [
    // [1.00 1.00 7.00 7.00]
    // [1.00 1.00 7.00 7.00]
    // [1.00 1.00 1.00 1.00]
    // ]


.. doxygenfunction:: set_select(const vector<string> &indices, Tensor *A)

Example:

.. code-block:: c++

   Tensor* t1 = Tensor::ones({3, 4});
   Tensor* t2 = Tensor::full({1, 1}, 5.0f);
   t1->set_select({"0", "2"}, t2); // Set element (0, 2) to "5"
   // [
   // [1.00 1.00 5.00 1.00]
   // [1.00 1.00 1.00 1.00]
   // [1.00 1.00 1.00 1.00]
   // ]

   Tensor* t3 = Tensor::ones({3, 4});
   Tensor* t4 = Tensor::full({3, 1}, 5.0f);
   t3->set_select({":", "2"}, t4);  // Set 2nd column to "5"
   // [
   // [1.00 1.00 5.00 1.00]
   // [1.00 1.00 5.00 1.00]
   // [1.00 1.00 5.00 1.00]
   // ]

   Tensor* t5 = Tensor::ones({3, 4});
   Tensor* t6 = Tensor::full({3, 2}, 5.0f);
   t5->set_select({"0:2", "-2:"}, t6);  // Set 3rd and 4th column from 1st and 2nd row to "5"
   // [
   // [1.00 1.00 5.00 5.00]
   // [1.00 1.00 5.00 5.00]
   // [1.00 1.00 1.00 1.00]
   // ]


Expand
^^^^^^^^^

.. doxygenfunction:: Tensor::Expand(int size)

Example:

.. code-block:: c++

    Tensor* t1 = new Tensor( {1, 2, 3}, {3, 1});

    Tensor* new_t = t1->expand(3);

   // Other ways
   Tensor::expand(t1, size); // static



Sorting
----------

sort
^^^^^^^^^

.. doxygenfunction:: Tensor::sort(bool descending = false, bool stable = true)


Example:

.. code-block:: c++

    Tensor* t1 = Tensor::randn({5});
   // [-0.01 0.34 0.10 -0.57 -0.28]

    Tensor* t2 = t1->sort();  // Ascending
   // [-0.57 -0.28 -0.01 0.10 0.34]

    Tensor* t3 = t1->sort(true);  // Descending==True
   // [0.34 0.10 -0.01 -0.28 -0.57]

   // Other ways
   t1->sort_();  // In-place
   Tensor::sort(t1, t2); // static


argsort
^^^^^^^^^

.. doxygenfunction:: Tensor::argsort(bool descending = false, bool stable = true)


Example:

.. code-block:: c++

    Tensor* t1 = Tensor::randn({5});
   // [-0.01 0.34 0.10 -0.57 -0.28]

    Tensor* t2 = t1->argsort();  // Ascending
   // [3 4 0 2 1]

    Tensor* t3 = t1->argsort(true);  // Descending==True
   // [1 2 0 4 3]

   // Other ways
   Tensor::argsort(t1, t2); // static




