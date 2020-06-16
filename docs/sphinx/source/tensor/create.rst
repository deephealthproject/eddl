Creation Routines
=================

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/eddl_progress_tensor.md


Constructors
-------------

Create an uninitialized tensor

.. doxygenfunction:: eddlT::create(const vector<int>&)
.. doxygenfunction:: eddlT::create(const vector<int>&, float *, int)
.. doxygenfunction:: eddlT::create(const vector<int>&, int)
.. doxygenfunction:: eddlT::create(const vector<int>&, float *)

.. code-block:: c++

    create(const vector<int> &shape);
    create(const vector<int> &shape, float *fptr, int dev=DEV_CPU);
    create(const vector<int> &shape, int dev=DEV_CPU);
    create(const vector<int> &shape, Tensor *tensor );

Constructors & Initializers
-----------------------------

Create tensor from generators

zeros
^^^^^^^^^

.. doxygenfunction:: eddlT::zeros

.. code-block:: c++

    static Tensor* zeros(const vector<int> &shape, int dev=DEV_CPU);
    
ones
^^^^^^^^^

.. doxygenfunction:: eddlT::ones

.. code-block:: c++

    static Tensor* ones(const vector<int> &shape, int dev=DEV_CPU);
    
full
^^^^^^^^^

.. doxygenfunction:: eddlT::full

.. code-block:: c++

    static Tensor* full(const vector<int> &shape, float value, int dev=DEV_CPU);


eye
^^^^^^^^^

.. doxygenfunction:: eddlT::eye

.. code-block:: c++

    static Tensor* eye(int rows, int offset=0, int dev=DEV_CPU);
    
identity
^^^^^^^^^

.. doxygenfunction:: eddlT::identity

.. code-block:: c++

    static Tensor* identity(int rows, int dev=DEV_CPU);
    // empty?




Constructors from existing data
--------------------------------

Create tensor from existing data

Move to CPU
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::toCPU_

.. doxygenfunction:: eddlT::toCPU

.. code-block:: c++

    void toCPU_(Tensor *A);
    Tensor* toCPU(Tensor *A);

Move to GPU
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::toGPU_

.. doxygenfunction:: eddlT::toGPU

.. code-block:: c++

    void toGPU_(Tensor *A);
    Tensor* toGPU(Tensor *A);

clone
^^^^^^^^^

.. doxygenfunction:: eddlT::clone

.. code-block:: c++

    Tensor* clone();
    

reallocate
^^^^^^^^^^^

.. doxygenfunction:: eddlT::reallocate

.. code-block:: c++

    void reallocate(Tensor* old_t, vector<int> *s = nullptr);
    

copy
^^^^^^^^^

.. doxygenfunction:: eddlT::copyTensor

.. code-block:: c++

    static void copy(Tensor *A, Tensor *B);
    //more

select
^^^^^^^^^

.. doxygenfunction:: eddlT::select

.. code-block:: c++

    Tensor* select(Tensor *A, int i);

Numerical ranges
-----------------

Create tensor from numerical ranges

arange
^^^^^^^^^

.. doxygenfunction:: eddlT::arange

.. code-block:: c++

    static Tensor* arange(float start, float end, float step=1.0f, int dev=DEV_CPU);
    
range
^^^^^^^^^

.. doxygenfunction:: eddlT::range

.. code-block:: c++

    static Tensor* range(float start, float end, float step=1.0f, int dev=DEV_CPU);
    
linspace
^^^^^^^^^

.. doxygenfunction:: eddlT::linspace

.. code-block:: c++

    static Tensor* linspace(float start, float end, int steps=100, int dev=DEV_CPU);
    
logspace
^^^^^^^^^

.. doxygenfunction:: eddlT::logspace

.. code-block:: c++

    static Tensor* logspace(float start, float end, int steps=100, float base=10.0f, int dev=DEV_CPU);
 

Random
-------

Create tensor from generators


randu
^^^^^^^^^

.. doxygenfunction:: eddlT::randu

.. code-block:: c++

    static Tensor* randu(const vector<int> &shape, int dev=DEV_CPU);
    
randn
^^^^^^^^^

.. doxygenfunction:: eddlT::randn

.. code-block:: c++

    Tensor* randn(const vector<int> &shape, int dev=DEV_CPU);


Build matrices
-----------------

.. doxygenfunction:: eddlT::diag(Tensor *, int, int)


Example:

.. code-block:: c++
   :linenos:

    static Tensor* diag(Tensor* A, int k=0, int dev=DEV_CPU);
    // tri?
