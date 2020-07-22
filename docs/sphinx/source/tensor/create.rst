Creation Routines
=================

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/docs/markdown/eddl_progress_tensor.md


Constructors
-------------

Create an uninitialized tensor

.. doxygenfunction:: Tensor::Tensor()
.. doxygenfunction:: Tensor::Tensor(const vector<int>&, int)
.. doxygenfunction:: Tensor::Tensor(const vector<int>&, float*, int)
.. doxygenfunction:: Tensor::Tensor(const vector<int>&, Tensor*)
.. doxygenfunction:: Tensor::Tensor(const vector<float>&, const vector<int>&, int)

.. code-block:: c++

    Tensor();    
    Tensor* tensor1 = Tensor(const vector<int> &shape, int dev=DEV_CPU);
    Tensor(const vector<int> &shape, float *fptr, int dev);
    Tensor(const vector<int> &shape, Tensor*T);
    Tensor(const vector<float>& data, const vector<int> &shape, int dev=DEV_CPU);



Constructors & Initializers
-----------------------------

Create tensor from generators

empty
^^^^^^^^^

.. doxygenfunction:: Tensor::empty

.. code-block:: c++

    vector<int> v {256,3} // Desired shape
    Tensor* tensor1 = Tensor::empty(v, DEV_CPU); //Creates an empty tensor of 256x3 in CPU

empty_like
^^^^^^^^^^^

.. doxygenfunction:: Tensor::empty_like

.. code-block:: c++

    vector<int> v {256,3} // Desired shape
    Tensor* tensor1 = Tensor::empty(v, DEV_CPU); // Tensor of 256x3 in CPU
    Tensor* tensor2 = Tensor::empty_like(tensor1); // Empty tensor taking shape and device from tensor1
    

zeros
^^^^^^^^^

.. doxygenfunction:: Tensor::zeros

.. code-block:: c++

    vector<int> v {3} // Desired shape
    Tensor* tensor1 = Tensor::zeros(v, DEV_CPU); // Creates 1D tensor filled with zeros
    //tensor1 => [0,0,0]

zeros_like
^^^^^^^^^^^

.. doxygenfunction:: Tensor::empty_like

.. code-block:: c++

    vector<int> v {3} // Desired shape
    Tensor* tensor1 = Tensor::empty(v, DEV_CPU); // Tensor of 3 components in CPU
    Tensor* tensor2 = Tensor::zeros_like(tensor1); // Tensor of 3 components in CPU filled with zeros
    // tensor2 => [0,0,0]
    
ones
^^^^^^^^^

.. doxygenfunction:: Tensor::ones

.. code-block:: c++

    vector<int> v {3} // Desired shape
    Tensor* tensor1 = Tensor::ones(v, DEV_CPU); // Creates 1D tensor filled with ones
    //tensor1 => [1,1,1]

ones_like
^^^^^^^^^^^

.. doxygenfunction:: Tensor::ones_like

.. code-block:: c++

    vector<int> v {3} // Desired shape
    Tensor* tensor1 = Tensor::empty(v, DEV_CPU); // Tensor of 3 components in CPU
    Tensor* tensor2 = Tensor::ones_like(tensor1); // Tensor of 3 components in CPU filled with zeros
    // tensor2 => [1,1,1]
    
full
^^^^^^^^^

.. doxygenfunction:: Tensor::full

.. code-block:: c++

    vector<int> v {3} // Desired shape
    Tensor* tensor1 = Tensor::full(v, 10, DEV_CPU); // Creates 1D tensor filled with 10s
    //tensor1 => [10,10,10]

full_like
^^^^^^^^^^^

.. doxygenfunction:: Tensor::full_like

.. code-block:: c++

    vector<int> v {3} // Desired shape
    Tensor* tensor1 = Tensor::empty(v, DEV_CPU); // Tensor of 3 components in CPU
    Tensor* tensor2 = Tensor::full_like(tensor1, 10); // Tensor of 3 components in CPU filled with 10s
    // tensor2 => [10,10,10]


eye
^^^^^^^^^

.. doxygenfunction:: Tensor::eye

.. code-block:: c++

    
    Tensor* matrix1 = Tensor::eye(3, 3, DEV_CPU);
    // matrix1 => [1 3 3
    //             3 1 3
    //             3 3 1]
    
identity
^^^^^^^^^

.. doxygenfunction:: Tensor::identity

.. code-block:: c++

    Tensor* matrix1 = Tensor::identity(3, DEV_CPU);
    // matrix1 => [1 0 0
    //             0 1 0
    //             0 0 1]




Constructors from existing data
--------------------------------

Create tensor from existing data

clone
^^^^^^^^^

.. doxygenfunction:: Tensor::clone

.. code-block:: c++

    vector<int> v {3} // Desired shape
    Tensor* tensor1 = Tensor::ones(v, DEV_CPU); // Creates 1D tensor filled with ones
    //tensor1 => [1,1,1]
    Tensor* tensor2 = tensor1->clone();
    //tensor2 => [1,1,1]
    

reallocate
^^^^^^^^^^^

.. doxygenfunction:: Tensor::reallocate

.. code-block:: c++

    void reallocate(Tensor* old_t, vector<int> *s = nullptr);
    

copy
^^^^^^^^^

.. doxygenfunction:: Tensor::copy

.. code-block:: c++

    vector<int> v {3} // Desired shape
    Tensor* tensor1 = Tensor::ones(v, DEV_CPU); // Creates 1D tensor filled with ones
    //tensor1 => [1,1,1]
    Tensor* tensor2;
    Tensor::copy(tensor1, tensor2);
    //tensor2 => [1,1,1]



Constructors from numerical ranges
------------------------------------

Create tensor from numerical ranges

arange
^^^^^^^^^

.. doxygenfunction:: Tensor::arange

.. code-block:: c++

    Tensor* tensor1 = Tensor::arange(1.0, 4.0, 0.5, DEV_CPU);
    // tensor1 => [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    
range
^^^^^^^^^

.. doxygenfunction:: Tensor::range

.. code-block:: c++

    Tensor* tensor1 = Tensor::range(1.0, 4.0, 0.5, DEV_CPU);
    // tensor1 => [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
linspace
^^^^^^^^^

.. doxygenfunction:: Tensor::linspace

.. code-block:: c++

    Tensor* tensor1 = Tensor::linspace(3.0, 10.0, 5, DEV_CPU);
    //tensor1 => [3.00, 4.75, 6.50, 8.25, 10.00]
    
logspace
^^^^^^^^^

.. doxygenfunction:: Tensor::logspace

.. code-block:: c++

    Tensor* tensor1 = Tensor::logspace(0.1, 1.0, 5, 10.0, DEV_CPU);
    //tensor1 => [1.2589, 2.1135, 3.5481, 5.9566, 10.0000]
 

geomspace
^^^^^^^^^^^

.. doxygenfunction:: Tensor::geomspace

.. code-block:: c++

    Tensor* tensor1 = Tensor::geomspace(1.0, 1000.0, 3, DEV_CPU);
    //tensor1 => [1.0, 10.0, 100.0]
 

Constructors from random generators
-------------------------------------

Create tensor from generators


randu
^^^^^^^^^

.. doxygenfunction:: Tensor::randu

.. code-block:: c++

    static Tensor* randu(const vector<int> &shape, int dev=DEV_CPU);
    
randn
^^^^^^^^^

.. doxygenfunction:: Tensor::randn

.. code-block:: c++

    Tensor* randn(const vector<int> &shape, int dev=DEV_CPU);


Constructors of matrices
-------------------------

.. doxygenfunction:: Tensor::diag_(int)
.. doxygenfunction:: Tensor::diag(int)
.. doxygenfunction:: Tensor::diag(Tensor*, Tensor*, int)


Example:

.. code-block:: c++
   :linenos:

    Tensor* matrix1 = Tensor::eye(3, 3, DEV_CPU);
    // matrix1 => [1 3 3
    //             3 1 3
    //             3 3 1]

    
    Tensor* main_diag = matrix1->diag(0);
    // main_diag => [1,1,1]

    Tensor* main_diag_2;
    Tensor::diag(matrix1, main_diag_2, 0);
    // main_diag_2 => [1,1,1]

    matrix1->diag_(0);
    // matrix1 => [1,1,1]


Destructors
-------------

Delete a tensor to free memory

.. doxygenfunction:: Tensor::~Tensor()

.. code-block:: c++

    ~Tensor();
