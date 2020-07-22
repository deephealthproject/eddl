Indexing & Sorting
=================

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/docs/markdown/eddl_progress_tensor.md

Indexing
--------------

nonzero
^^^^^^^^^

.. doxygenfunction:: Tensor::nonzero

Example:

.. code-block:: c++
   :linenos:

    Tensor* t1 = new Tensor::Tensor({0,10,20,0,50,0}, {6}, DEV_CPU);
    Tensor* indices = t1->nonzero(true); //Return, sorted, the indices whose values are not zero
    // indices => [1,2,4]

where
^^^^^^^^^

.. doxygenfunction:: Tensor::where(Tensor*, Tensor*, Tensor *)
.. doxygenfunction:: Tensor::where(Tensor*, Tensor*, Tensor*, Tensor *)


Example:

.. code-block:: c++
   :linenos:

    static Tensor* where(Tensor *condition, Tensor *A, Tensor *B);  // where(x > 0, x[random], y[ones])
    static void where(Tensor *condition, Tensor *A, Tensor *B, Tensor *C);


select
^^^^^^^^^

.. doxygenfunction:: Tensor::select(Tensor*, Tensor*, vector<int>, int, int, bool)


Example:

.. code-block:: c++
   :linenos:

    static void select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end, bool mask_zeros=false);


deselect
^^^^^^^^^

.. doxygenfunction:: Tensor::deselect(Tensor*, Tensor*, vector<int>, int, int, int, bool)


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
.. doxygenfunction:: Tensor::sort(Tensor*, Tensor*, bool, bool)


Example:

.. code-block:: c++
   :linenos:

    Tensor* t1 = new Tensor::Tensor({100,90,0,50,3,1}, {6}, DEV_CPU);

    Tensor* sorted1 = t1->sort(); //Sort ascending
    // sorted1 => [0,1,3,50,90,100]

    Tensor* sorted2;
    Tensor::sort(t1, sorted2, true); // Sort descending
    // sorted2 => [100,90,50,3,1,0]

    t1->sort_();//Sort ascending inplace
    // t1 => [0,1,3,50,90,100]
    


argsort
^^^^^^^^^

.. doxygenfunction:: Tensor::argsort(bool, bool)
.. doxygenfunction:: Tensor::argsort(Tensor*, Tensor*, bool, bool)


Example:

.. code-block:: c++
   :linenos:

    Tensor* t1 = new Tensor::Tensor({100,90,0,50,3,1}, {6}, DEV_CPU);

    Tensor* sorted_indices = t1->argsort(); //Sort indices ascending
    // sorted_indices => [2,5,4,3,1,0]

    Tensor* sorted_indices2;
    Tensor::argsort(t1, sorted_indices2, true); //Sort indices descending
    // sorted_indices => [0,1,3,4,5,2]




