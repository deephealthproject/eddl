Creation Routines
=================

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/eddl_progress_tensor.md


Constructor
------------

Create an uninitialized tensor

.. doxygenfunction:: Tensor::Tensor( const vector<int>&, float *, int)

.. code-block:: c++

    Tensor(const vector<int> &shape, float *fptr, int dev=DEV_CPU);


Ones and zeros
--------------

Create tensor from generators

zeros
^^^^^^^^^

.. doxygenfunction:: Tensor::zeros

Create a tensor of the specified shape and fill it with zeros.

  Parameters:

  - ``&shape``: Shape of the tensor to create.
  - ``dev``: Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.

.. code-block:: c++

    static Tensor* zeros(const vector<int> &shape, int dev=DEV_CPU);
    
ones
^^^^^^^^^

.. doxygenfunction:: Tensor::ones

Create a tensor of the specified shape and fill it with ones.
  
  Parameters:

  - ``&shape``: Shape of the tensor to create.
  - ``dev``: Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.

.. code-block:: c++

    static Tensor* ones(const vector<int> &shape, int dev=DEV_CPU);
    
full
^^^^^^^^^

.. doxygenfunction:: Tensor::full

Create a tensor of the specified shape and fill it with the value ``value``.

  Parameters:

  - ``&shape``: Shape of the tensor to create.
  - ``value``: Value to use to fill the tensor.
  - ``dev``: Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.

.. code-block:: c++

    static Tensor* full(const vector<int> &shape, float value, int dev=DEV_CPU);


eye
^^^^^^^^^

.. doxygenfunction:: Tensor::eye


Parameters:

  - ``rows``: Number of rows of the tensor.
  - ``offset``:
  - ``dev``: Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.

.. code-block:: c++

    static Tensor* eye(int rows, int offset=0, int dev=DEV_CPU);
    
identity
^^^^^^^^^

.. doxygenfunction:: Tensor::identity

Create a tensor representing the identity matrix. Equivalent to calling function ``eye`` with ``offset = 0``.

  Parameters:

  - ``rows``: Number of rows of the tensor.
  - ``dev``: Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.

.. code-block:: c++

    static Tensor* identity(int rows, int dev=DEV_CPU);
    // empty?


From existing data
-------------------

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


Numerical ranges
-----------------

Create tensor from numerical ranges

arange
^^^^^^^^^

.. doxygenfunction:: Tensor::arange(float, float, float, int)

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
^^^^^^^^^^

.. doxygenfunction:: Tensor::geomspace

.. code-block:: c++

    static Tensor* geomspace(float start, float end, int steps=100, int dev=DEV_CPU);


Random
-------

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

    static Tensor* randn(const vector<int> &shape, int dev=DEV_CPU);


Build matrices
-----------------

.. doxygenfunction:: Tensor::diag(Tensor *, int, int)

Create tensor from generators

Example:

.. code-block:: c++
   :linenos:

    static Tensor* diag(Tensor* A, int k=0, int dev=DEV_CPU);
    // tri?
