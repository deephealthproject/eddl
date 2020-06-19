Creation Routines
=================

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/eddl_progress_tensor.md


Constructors
-------------

Create an uninitialized tensor

.. doxygenfunction:: Tensor::Tensor()
.. doxygenfunction:: Tensor::Tensor(const vector<int>&, int)
.. doxygenfunction:: Tensor::Tensor(const vector<int>&, float *, int)
.. doxygenfunction:: Tensor::Tensor(const vector<int>&, Tensor *)
.. doxygenfunction:: Tensor::Tensor(const vector<float>&, const vector<int>&, int)

.. code-block:: c++

    Tensor();
    Tensor(const vector<int> &shape, int dev=DEV_CPU);
    Tensor(const vector<int> &shape, float *fptr, int dev);
    Tensor(const vector<int> &shape, Tensor *T);
    Tensor(const vector<float>& data, const vector<int> &shape, int dev=DEV_CPU);



Constructors & Initializers
-----------------------------

Create tensor from generators

empty
^^^^^^^^^

.. doxygenfunction:: Tensor::empty

.. code-block:: c++

    static Tensor* empty(const vector<int> &shape, int dev=DEV_CPU);

empty_like
^^^^^^^^^^^

.. doxygenfunction:: Tensor::empty_like

.. code-block:: c++

    static Tensor* empty_like(Tensor *A);
    

zeros
^^^^^^^^^

.. doxygenfunction:: Tensor::zeros

.. code-block:: c++

    static Tensor* zeros(const vector<int> &shape, int dev=DEV_CPU);

zeros_like
^^^^^^^^^^^

.. doxygenfunction:: Tensor::empty_like

.. code-block:: c++

    static Tensor* zeros_like(Tensor *A);
    
ones
^^^^^^^^^

.. doxygenfunction:: Tensor::ones

.. code-block:: c++

    static Tensor* ones(const vector<int> &shape, int dev=DEV_CPU);

ones_like
^^^^^^^^^^^

.. doxygenfunction:: Tensor::ones_like

.. code-block:: c++

    static Tensor* ones_like(Tensor *A);
    
full
^^^^^^^^^

.. doxygenfunction:: Tensor::full

.. code-block:: c++

    static Tensor* full(const vector<int> &shape, float value, int dev=DEV_CPU);

full_like
^^^^^^^^^^^

.. doxygenfunction:: Tensor::full_like

.. code-block:: c++

    static Tensor* full_like(Tensor *A);


eye
^^^^^^^^^

.. doxygenfunction:: Tensor::eye

.. code-block:: c++

    static Tensor* eye(int rows, int offset=0, int dev=DEV_CPU);
    
identity
^^^^^^^^^

.. doxygenfunction:: Tensor::identity

.. code-block:: c++

    static Tensor* identity(int rows, int dev=DEV_CPU);
    // empty?




Constructors from existing data
--------------------------------

Create tensor from existing data

clone
^^^^^^^^^

.. doxygenfunction:: Tensor::clone

.. code-block:: c++

    Tensor* clone();
    

reallocate
^^^^^^^^^^^

.. doxygenfunction:: Tensor::reallocate

.. code-block:: c++

    void reallocate(Tensor* old_t, vector<int> *s = nullptr);
    

copy
^^^^^^^^^

.. doxygenfunction:: Tensor::copy

.. code-block:: c++

    static void copy(Tensor *A, Tensor *B);
    //more

select
^^^^^^^^^

.. doxygenfunction:: Tensor::select(const vector<string>&)
.. doxygenfunction:: Tensor::select(Tensor *, Tensor *, SelDescriptor *)
.. doxygenfunction:: Tensor::select(Tensor *, Tensor *, vector<int>, int, int, bool)

.. code-block:: c++

    Tensor* select(const vector<string>& indices);
    static void select(Tensor *A, Tensor *B, SelDescriptor *sd);
    static void select_back(Tensor *A, Tensor *B, SelDescriptor *sd);


Constructors from numerical ranges
------------------------------------

Create tensor from numerical ranges

arange
^^^^^^^^^

.. doxygenfunction:: Tensor::arange

.. code-block:: c++

    static Tensor* arange(float start, float end, float step=1.0f, int dev=DEV_CPU);
    
range
^^^^^^^^^

.. doxygenfunction:: Tensor::range

.. code-block:: c++

    static Tensor* range(float start, float end, float step=1.0f, int dev=DEV_CPU);
    
linspace
^^^^^^^^^

.. doxygenfunction:: Tensor::linspace

.. code-block:: c++

    static Tensor* linspace(float start, float end, int steps=100, int dev=DEV_CPU);
    
logspace
^^^^^^^^^

.. doxygenfunction:: Tensor::logspace

.. code-block:: c++

    static Tensor* logspace(float start, float end, int steps=100, float base=10.0f, int dev=DEV_CPU);
 

geomspace
^^^^^^^^^^^

.. doxygenfunction:: Tensor::geomspace

.. code-block:: c++

    static Tensor* geomspace(float start, float end, int steps=100, int dev=DEV_CPU);
 

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
.. doxygenfunction:: Tensor::diag(Tensor *, Tensor *, int)


Example:

.. code-block:: c++
   :linenos:

    void diag_(int k=0);
    Tensor* diag(int k=0);
    static void diag(Tensor* A, Tensor* B, int k=0);


Destructors
-------------

Delete a tensor to free memory

.. doxygenfunction:: Tensor::~Tensor()

.. code-block:: c++

    ~Tensor();
