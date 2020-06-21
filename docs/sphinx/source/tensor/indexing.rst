Indexing & Sorting
=================

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/eddl_progress_tensor.md

Indexing
--------------

nonzero
^^^^^^^^^

.. doxygenfunction:: Tensor::nonzero

Example:

.. code-block:: c++
   :linenos:

    Tensor* nonzero(bool sort_indices=false);


where
^^^^^^^^^

.. doxygenfunction:: Tensor::where(Tensor *, Tensor *, Tensor *)
.. doxygenfunction:: Tensor::where(Tensor *, Tensor *, Tensor *, Tensor *)


Example:

.. code-block:: c++
   :linenos:

    static Tensor* where(Tensor *condition, Tensor *A, Tensor *B);  // where(x > 0, x[random], y[ones])
    static void where(Tensor *condition, Tensor *A, Tensor *B, Tensor *C);


select
^^^^^^^^^

.. doxygenfunction:: Tensor::select(Tensor *, Tensor *, vector<int>, int, int, bool)


Example:

.. code-block:: c++
   :linenos:

    static void select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end, bool mask_zeros=false);


deselect
^^^^^^^^^

.. doxygenfunction:: Tensor::deselect(Tensor *, Tensor *, vector<int>, int, int, int, bool)


Example:

.. code-block:: c++
   :linenos:

    static void deselect(Tensor *A, Tensor *B, vector<int> sind, int ini, int end,int inc=0, bool mask_zeros=false);



Sorting
----------

sort
^^^^^^^^^

.. doxygenfunction:: Tensor::sort_(bool, bool)
.. doxygenfunction:: Tensor::sort(bool, bool)
.. doxygenfunction:: Tensor::sort(Tensor *, Tensor *, bool, bool)


Example:

.. code-block:: c++
   :linenos:

    void sort_(bool descending=false, bool stable=true);
    Tensor* sort(bool descending=false, bool stable=true);
    static void sort(Tensor* A, Tensor* B, bool descending=false, bool stable=true);


argsort
^^^^^^^^^

.. doxygenfunction:: Tensor::argsort(bool, bool)
.. doxygenfunction:: Tensor::argsort(Tensor *, Tensor *, bool, bool)


Example:

.. code-block:: c++
   :linenos:

    void argsort(bool descending=false, bool stable=true);
    Tensor* argsort(bool descending=false, bool stable=true);




