Logic functions
===============

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/eddl_progress_tensor.md


Truth value testing
---------------------------


all
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::all

.. code-block:: c++

    static bool all(Tensor *A);
    

any
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::any

.. code-block:: c++

    static bool any(Tensor *A);


Array contents
-----------------



isfinite
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isfinite

.. code-block:: c++

    static void isfinite(Tensor *A, Tensor* B);
    

isinf
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isinf

.. code-block:: c++

    static void isinf(Tensor *A, Tensor* B);
    

isnan
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isnan

.. code-block:: c++

    static void isnan(Tensor *A, Tensor* B);
    

isneginf
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isneginf

.. code-block:: c++

    static void isneginf(Tensor *A, Tensor* B);
    

isposinf
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isposinf

.. code-block:: c++

    static void isposinf(Tensor *A, Tensor* B);



Logical operations
---------------------------


logical_and
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logical_and

.. code-block:: c++

    static void logical_and(Tensor *A, Tensor *B, Tensor *C);
        

logical_or
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logical_or

.. code-block:: c++

    static void logical_or(Tensor *A, Tensor *B, Tensor *C);
        

logical_not
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logical_not

.. code-block:: c++

    static void logical_not(Tensor *A, Tensor *B);
        

logical_xor
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logical_xor

.. code-block:: c++

    static void logical_xor(Tensor *A, Tensor *B, Tensor *C);



Comparison
---------------------------


allclose
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::allclose

.. code-block:: c++

    static bool allclose(Tensor *A, Tensor *B, float rtol=1e-05, float atol=1e-08, bool equal_nan=false);  // Returns true or false
    

isclose
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isclose

.. code-block:: c++

    static void isclose(Tensor *A, Tensor *B, Tensor *C, float rtol=1e-05, float atol=1e-08, bool equal_nan=false);  // Returns a boolean tensor
        

greater
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::greater
 
.. code-block:: c++

    static void greater(Tensor *A, Tensor *B, Tensor *C);


greater_equal
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::greater_equal

.. code-block:: c++

    static void greater_equal(Tensor *A, Tensor *B, Tensor *C);


less
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::less

.. code-block:: c++

    static void less(Tensor *A, Tensor *B, Tensor *C);


less_equal
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::less_equal

.. code-block:: c++

    static void less_equal(Tensor *A, Tensor *B, Tensor *C);


equal
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::equal

.. code-block:: c++

    static void equal(Tensor *A, Tensor *B, Tensor *C);
        

not_equal
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::not_equal

.. code-block:: c++

    static void not_equal(Tensor *A, Tensor *B, Tensor *C);