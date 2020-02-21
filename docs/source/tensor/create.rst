Creation Routines
=================

.. note::

    Section in progress

    Read this: <https://github.com/deephealthproject/eddl/blob/master/eddl_progress_tensor.md>


Constructor
------------

Create an uninitialized tensor

Example:

.. code-block:: c++
   :linenos:

    Tensor(const vector<int> &shape, float *fptr, int dev=DEV_CPU);


Ones and zeros
--------------

Create tensor from generators

Example:

.. code-block:: c++
   :linenos:

    static Tensor* zeros(const vector<int> &shape, int dev=DEV_CPU);
    static Tensor* ones(const vector<int> &shape, int dev=DEV_CPU);
    static Tensor* full(const vector<int> &shape, float value, int dev=DEV_CPU);
    static Tensor* eye(int rows, int offset=0, int dev=DEV_CPU);
    static Tensor* identity(int rows, int dev=DEV_CPU);
    // empty?


From existing data
-------------------

Create tensor from existing data

Example:

.. code-block:: c++
   :linenos:

    Tensor* clone();
    void reallocate(Tensor* old_t, vector<int> *s = nullptr);
    //more


Numerical ranges
-----------------

Create tensor from numerical ranges

Example:

.. code-block:: c++
   :linenos:

    static Tensor* arange(float start, float end, float step=1.0f, int dev=DEV_CPU);
    static Tensor* range(float start, float end, float step=1.0f, int dev=DEV_CPU);
    static Tensor* linspace(float start, float end, int steps=100, int dev=DEV_CPU);
    static Tensor* logspace(float start, float end, int steps=100, float base=10.0f, int dev=DEV_CPU);
    static Tensor* geomspace(float start, float end, int steps=100, int dev=DEV_CPU);


Random
-------

Create tensor from generators

Example:

.. code-block:: c++
   :linenos:

    static Tensor* randu(const vector<int> &shape, int dev=DEV_CPU);
    static Tensor* randn(const vector<int> &shape, int dev=DEV_CPU);


Build matrices
-----------------

Create tensor from generators

Example:

.. code-block:: c++
   :linenos:

    static Tensor* diag(Tensor* A, int k=0, int dev=DEV_CPU);
    // tri?
