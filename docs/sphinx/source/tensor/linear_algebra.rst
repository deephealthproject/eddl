Linear algebra
==============


Matrix and vector operations
-------------------------------

Interpolate
^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::interpolate(float factor1, Tensor *A, float factor2, Tensor *B)

Example:

.. code-block:: c++

   Tensor* t1 = Tensor::full({2, 3}, 1.0f);
   // [
   // [1.00 1.00 1.00]
   // [1.00 1.00 1.00]
   // ]

   Tensor* t2 = Tensor::full({2, 3}, 10.0f);
   // [
   // [10.00 10.00 10.00]
   // [10.00 10.00 10.00]
   // ]

    Tensor* t3 = Tensor::interpolate(2.5f, t1, 0.5f, t2);  // a*t1 + b*t2
   // [
   // [7.50 7.50 7.50]
   // [7.50 7.50 7.50]
   // ]


Trace
^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::trace(int k = 0)

Example:

.. code-block:: c++

   Tensor* t1 = new Tensor({1, 3, 5, 4, 1, 3, 5, 4, 1}, {3, 3});
   // [
   // [1.00 3.00 5.00]
   // [4.00 1.00 3.00]
   // [5.00 4.00 1.00]
   // ]

   float tr0 = t1->trace(0);
   // 3

   float tr1 = t1->trace(1);
   // 6


Norm
^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::norm(string ord = "fro")
.. doxygenfunction:: Tensor::norm(vector<int> axis, bool keepdims, string ord = "fro")

Example:

.. code-block:: c++

   Tensor* t1 = new Tensor({1,2,3,4,5,6}, {3, 2});
   // [
   // [1.00 2.00 3.00]
   // [4.00 5.00 6.00]
   // ]

   // Global (reduce on all axis)
   float n1 = t1->norm();
   // 9.53939

   // Reduced on axis 0
   Tensor* t2 = t1->norm({0}, false); // keepdims==false
   // [4.12 5.39 6.71]

   // Other ways
   Tensor::norm(t1, "fro");

