Creation Routines
=================


Constructors
-------------

Create an uninitialized tensor

.. doxygenfunction:: Tensor::Tensor()

.. code-block:: c++

   Tensor* t1 = new Tensor();
   // []


.. doxygenfunction:: Tensor::Tensor(const vector<int> &shape, int dev = DEV_CPU)

.. code-block:: c++

   Tensor* t1 = new Tensor({2, 3});
   // [
   // [0.00 -0.00 0.00]
   // [-0.00 0.00 0.00]
   // ]


.. doxygenfunction:: Tensor::Tensor(const vector<float> &data, const vector<int> &shape, int dev = DEV_CPU)

.. code-block:: c++

   Tensor* t1 = new Tensor({1,2,3,4,5,6}, {2,3});
   // [
   // [1.00 2.00 3.00]
   // [4.00 5.00 6.00]
   // ]



Constructors & Initializers
-----------------------------

Create tensor from generators

empty
^^^^^^^^^

.. doxygenfunction:: Tensor::empty

.. code-block:: c++

   Tensor* t1 = Tensor::empty({2, 3});
   // [
   // [0.00 -36893488147419103232.00 0.00]
   // [-36893488147419103232.00 0.00 0.00]
   // ]


empty_like
^^^^^^^^^^^

.. doxygenfunction:: Tensor::empty_like

.. code-block:: c++

   Tensor* t1 = Tensor::empty({2, 3});
   Tensor* t2 = Tensor::empty_like(t1);
   // [
   // [0.00 -36893488147419103232.00 0.00]
   // [-36893488147419103232.00 0.00 0.00]
   // ]


zeros
^^^^^^^^^

.. doxygenfunction:: Tensor::zeros

.. code-block:: c++

   Tensor* t1 = Tensor::zeros({2, 3});
   // [
   // [0.00 0.00 0.00]
   // [0.00 0.00 0.00]
   // ]


zeros_like
^^^^^^^^^^^

.. doxygenfunction:: Tensor::zeros_like

.. code-block:: c++

    Tensor* t1 = Tensor::empty({2, 3});
    Tensor* t2 = Tensor::zeros_like(t1);
   // [
   // [0.00 0.00 0.00]
   // [0.00 0.00 0.00]
   // ]


ones
^^^^^^^^^

.. doxygenfunction:: Tensor::ones

.. code-block:: c++

   Tensor* t1 = Tensor::ones({2, 3});
   // [
   // [1.00 1.00 1.00]
   // [1.00 1.00 1.00]
   // ]


ones_like
^^^^^^^^^^^

.. doxygenfunction:: Tensor::ones_like

.. code-block:: c++

   Tensor* t1 = Tensor::empty({2, 3});
   Tensor* t2 = Tensor::ones_like(t1);
   // [
   // [1.00 1.00 1.00]
   // [1.00 1.00 1.00]
   // ]


full
^^^^^^^^^

.. doxygenfunction:: Tensor::full

.. code-block:: c++

   Tensor* t1 = Tensor::full({2, 3}, 10.0f);
   // [
   // [10.00 10.00 10.00]
   // [10.00 10.00 10.00]
   // ]


full_like
^^^^^^^^^^^

.. doxygenfunction:: Tensor::full_like

.. code-block:: c++

   Tensor* t1 = Tensor::empty({2, 3});
   Tensor* t2 = Tensor::full_like(t1, 10.0f);
   // [
   // [10.00 10.00 10.00]
   // [10.00 10.00 10.00]
   // ]

eye
^^^^^^^^^

.. doxygenfunction:: Tensor::eye

.. code-block:: c++


   Tensor* t1 = Tensor::eye(3);
   // [
   // [1.00 0.00 0.00]
   // [0.00 1.00 0.00]
   // [0.00 0.00 1.00]
   // ]


   Tensor* t1 = Tensor::eye(3, -1);
   // [
   // [0.00 0.00 0.00]
   // [1.00 0.00 0.00]
   // [0.00 1.00 0.00]
   // ]


identity
^^^^^^^^^

.. doxygenfunction:: Tensor::identity

.. code-block:: c++

    Tensor* t1 = Tensor::identity(3);
   // [
   // [1.00 0.00 0.00]
   // [0.00 1.00 0.00]
   // [0.00 0.00 1.00]
   // ]



Constructors from existing data
--------------------------------

Create tensor from existing data

clone
^^^^^^^^^

.. doxygenfunction:: Tensor::clone

.. code-block:: c++

   Tensor* tensor1 = Tensor::ones({2, 3});
   // [
   // [1.00 1.00 1.00]
   // [1.00 1.00 1.00]
   // ]

   Tensor* tensor2 = tensor1->clone();
   // [
   // [1.00 1.00 1.00]
   // [1.00 1.00 1.00]
   // ]


reallocate
^^^^^^^^^^^

.. doxygenfunction:: Tensor::reallocate(Tensor *old_t, const vector<int> &shape)

.. code-block:: c++

    Tensor* t1 = new Tensor({1,2,3,4,5,6,7,8,9}, {3, 3});
    // [
    // [1.00 2.00 3.00]
    // [4.00 5.00 6.00]
    // [7.00 8.00 9.00]
    // ]

    Tensor* t2 = Tensor::zeros({1, 6});
    // [
    // [0.00 0.00 0.00 0.00 0.00 0.00]
    // ]

    t2->reallocate(t1);
    // [
    // [1.00 2.00 3.00 4.00 5.00 6.00]
    // ]

    t2->reallocate(t1, {2, 2});
    // [
    // [1.00 2.00]
    // [3.00 4.00]
    // ]

    t2->reallocate(t1, {3, 2});
    // [
    // [1.00 2.00]
    // [3.00 4.00]
    // [5.00 6.00]
    // ]

    // Modify value of T1
    t1->ptr[0] = 100.0f;

    // Tensor 1
    // [
    // [100.00 2.00 3.00]
    // [4.00 5.00 6.00]
    // [7.00 8.00 9.00]
    // ]

    // Tensor 2
    // [
    // [100.00 2.00]
    // [3.00 4.00]
    // [5.00 6.00]
    // ]

copy
^^^^^^^^^

.. doxygenfunction:: Tensor::copy

.. code-block:: c++

   Tensor* t1 = Tensor::ones({4, 3});
   // [
   // [1.00 1.00 1.00]
   // [1.00 1.00 1.00]
   // [1.00 1.00 1.00]
   // [1.00 1.00 1.00]
   // ]

   Tensor* t2 = Tensor::zeros({6, 2, 1});
   // [
   // [[0.00] [0.00]]
   // [[0.00] [0.00]]
   // [[0.00] [0.00]]
   // [[0.00] [0.00]]
   // [[0.00] [0.00]]
   // [[0.00] [0.00]]
   // ]

   Tensor::copy(t1, t2);
   // [
   // [[1.00] [1.00]]
   // [[1.00] [1.00]]
   // [[1.00] [1.00]]
   // [[1.00] [1.00]]
   // [[1.00] [1.00]]
   // [[1.00] [1.00]]
   // ]



Constructors from numerical ranges
------------------------------------

Create tensor from numerical ranges

arange
^^^^^^^^^

.. doxygenfunction:: Tensor::arange

.. code-block:: c++

    Tensor* t1 = Tensor::arange(1.0, 4.0, 0.5);
    // [1.00 1.50 2.00 2.50 3.00 3.50]

    
range
^^^^^^^^^

.. doxygenfunction:: Tensor::range

.. code-block:: c++

    Tensor* t1 = Tensor::range(1.0, 4.0, 0.5);
    // [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


linspace
^^^^^^^^^

.. doxygenfunction:: Tensor::linspace

.. code-block:: c++

    Tensor* t1 = Tensor::linspace(3.0, 10.0, 5);
    // [3.00, 4.75, 6.50, 8.25, 10.00]


logspace
^^^^^^^^^

.. doxygenfunction:: Tensor::logspace

.. code-block:: c++

    Tensor* t1 = Tensor::logspace(0.1, 1.0, 5, 10.0, DEV_CPU);
    // [1.2589, 2.1135, 3.5481, 5.9566, 10.0000]
 

geomspace
^^^^^^^^^^^

.. doxygenfunction:: Tensor::geomspace

.. code-block:: c++

    Tensor* t1 = Tensor::geomspace(1.0, 1000.0, 3, DEV_CPU);
    // [1.0, 10.0, 100.0]
 

Constructors from random generators
-------------------------------------

Create tensor from generators


randu
^^^^^^^^^

.. doxygenfunction:: Tensor::randu

.. code-block:: c++

   Tensor* t1 = Tensor::randu({2, 3});
   // [
   // [0.72 0.72 0.15]
   // [0.72 0.67 0.67]
   // ]


randn
^^^^^^^^^

.. doxygenfunction:: Tensor::randn

.. code-block:: c++

   Tensor* t1 = Tensor::randn({2, 3});
   // [
   // [-0.88 0.79 0.84]
   // [-0.64 0.35 0.04]
   // ]


Constructors of matrices
-------------------------

.. doxygenfunction:: Tensor::diag(int k = 0)

Example:

.. code-block:: c++

   Tensor* t1 = new Tensor({1,2,3,4,5,6,7,8,9}, {3, 3});
   // [
   // [1.00 2.00 3.00]
   // [4.00 5.00 6.00]
   // [7.00 8.00 9.00]
   // ]

   Tensor* t2 = t1->diag(0);
   // [
   // [1.00 0.00 0.00]
   // [0.00 5.00 0.00]
   // [0.00 0.00 9.00]
   // ]

   Tensor* t3 = t1->diag(1);
   // [
   // [0.00 2.00 0.00]
   // [0.00 0.00 6.00]
   // [0.00 0.00 0.00]
   // ]


