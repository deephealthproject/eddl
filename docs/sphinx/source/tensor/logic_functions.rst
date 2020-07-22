Logic functions
===============

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/docs/markdown/eddl_progress_tensor.md


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

    static void isfinite(Tensor *A, Tensor * B);
    

isinf
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isinf

.. code-block:: c++

    static void isinf(Tensor *A, Tensor * B);
    

isnan
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isnan

.. code-block:: c++

    static void isnan(Tensor *A, Tensor * B);
    

isneginf
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isneginf

.. code-block:: c++

    static void isneginf(Tensor *A, Tensor * B);
    

isposinf
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isposinf

.. code-block:: c++

    static void isposinf(Tensor *A, Tensor * B);



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

.. doxygenfunction:: Tensor::greater_(float)
.. doxygenfunction:: Tensor::greater(float)
.. doxygenfunction:: Tensor::greater(Tensor*, Tensor*, float)
.. doxygenfunction:: Tensor::greater(Tensor *)
.. doxygenfunction:: Tensor::greater(Tensor*, Tensor*, Tensor *)

 
.. code-block:: c++

    void greater_(float v);
    Tensor * greater(float v);
    static void greater(Tensor *A, Tensor *B, float v);
    Tensor * greater(Tensor *A);
    static void greater(Tensor *A, Tensor *B, Tensor *C);
    static void greater(Tensor *A, Tensor *B, Tensor *C);


greater_equal
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::greater_equal_(float)
.. doxygenfunction:: Tensor::greater_equal(float)
.. doxygenfunction:: Tensor::greater_equal(Tensor*, Tensor*, float)
.. doxygenfunction:: Tensor::greater_equal(Tensor *)
.. doxygenfunction:: Tensor::greater_equal(Tensor*, Tensor*, Tensor *)


.. code-block:: c++

    void greater_equal_(float v);
    Tensor * greater_equal(float v);
    static void greater_equal(Tensor *A, Tensor *B, float v);
    Tensor * greater_equal(Tensor *A);
    static void greater_equal(Tensor *A, Tensor *B, Tensor *C);


less
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::less_(float)
.. doxygenfunction:: Tensor::less(float)
.. doxygenfunction:: Tensor::less(Tensor*, Tensor*, float)
.. doxygenfunction:: Tensor::less(Tensor *)
.. doxygenfunction:: Tensor::less(Tensor*, Tensor*, Tensor *)

.. code-block:: c++

    void less_(float v);
    Tensor * less(float v);
    static void less(Tensor *A, Tensor *B, float v);
    Tensor * less(Tensor *A);
    static void less(Tensor *A, Tensor *B, Tensor *C);


less_equal
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::less_equal_(float)
.. doxygenfunction:: Tensor::less_equal(float)
.. doxygenfunction:: Tensor::less_equal(Tensor*, Tensor*, float)
.. doxygenfunction:: Tensor::less_equal(Tensor *)
.. doxygenfunction:: Tensor::less_equal(Tensor*, Tensor*, Tensor *)


.. code-block:: c++

    void less_equal_(float v);
    Tensor * less_equal(float v);
    static void less_equal(Tensor *A, Tensor *B, float v);
    Tensor * less_equal(Tensor *A);
    static void less_equal(Tensor *A, Tensor *B, Tensor *C);


equal
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::equal_(float)
.. doxygenfunction:: Tensor::equal(float)
.. doxygenfunction:: Tensor::equal(Tensor*, Tensor*, float)
.. doxygenfunction:: Tensor::equal(Tensor *)
.. doxygenfunction:: Tensor::equal(Tensor*, Tensor*, Tensor *)


.. code-block:: c++

    void equal_(float v);
    Tensor * equal(float v);
    static void equal(Tensor *A, Tensor *B, float v);
    Tensor * equal(Tensor *A);
    static void equal(Tensor *A, Tensor *B, Tensor *C);
        

not_equal
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::not_equal_(float)
.. doxygenfunction:: Tensor::not_equal(float)
.. doxygenfunction:: Tensor::not_equal(Tensor*, Tensor*, float)
.. doxygenfunction:: Tensor::not_equal(Tensor *)
.. doxygenfunction:: Tensor::not_equal(Tensor*, Tensor*, Tensor *)



.. code-block:: c++

    void not_equal_(float v);
    Tensor * not_equal(float v);
    static void not_equal(Tensor *A, Tensor *B, float v);
    Tensor * not_equal(Tensor *A);
    static void not_equal(Tensor *A, Tensor *B, Tensor *C);
