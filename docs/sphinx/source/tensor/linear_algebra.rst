Linear algebra
==============

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/docs/markdown/eddl_progress_tensor.md


Matrix and vector operations
-------------------------------

interpolate
^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::interpolate(float, Tensor*, float, Tensor*)
.. doxygenfunction:: Tensor::interpolate(float, Tensor*, float, Tensor*, Tensor*)

Example:

.. code-block:: c++
   :linenos:

    Tensor* t1 = new Tensor::Tensor({1,2,3,4,5,6}, {6}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({0,1,2,3,4,5}, {6}, DEV_CPU);
    Tensor* t3;

    Tensor* result = Tensor::interpolate(0.5, t1, 0.6, t2); // (new)result = f1*A + f2*B
    // result => [0.5, 1.6, 2.7, 3.8, 4.9, 6]

    Tensor::interpolate(0.5, t1, 0,6, t2, t3);  // t3 = f1*A + f2*B
    // t3 => [0.5, 1.6, 2.7, 3.8, 4.9, 6]


trace
^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::trace(int)
.. doxygenfunction:: Tensor::trace(Tensor*, int)

Example:

.. code-block:: c++
   :linenos:

    Tensor* matrix1 = Tensor::eye(3, 3, DEV_CPU);
    // matrix1 => [1 3 3
    //             3 1 3
    //             3 3 1]

    float tr =  matrix1->trace(0); // tr = 3
    float tr2 = Tensor::trace(matrix1, 0); // tr2 = 3


norm
^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::norm(string)
.. doxygenfunction:: Tensor::norm(Tensor*, string)
.. doxygenfunction:: Tensor::norm(vector<int>, bool, string)

Example:

.. code-block:: c++
   :linenos:

    Tensor* matrix1 = Tensor::eye(3, 3, DEV_CPU);
    // matrix1 => [1 3 3
    //             3 1 3
    //             3 3 1]

    Tensor* t1 = new Tensor::Tensor({1,2,3,4,5,6}, {6}, DEV_CPU);


    float m_norm = matrix1->norm(); //Frobenius norm of matrix1
    // m_norm => 7.5498

    float t_norm = Tensor::norm(t1); //Frobenius norm of t1
    // t_norm => 9.5394

    Tensor* m_norm2 = matrix1->norm({0}, false);//Frobenius norm over rows in matrix1
    // m_norm2 => [4.3589, 4.3589, 4.3589]


