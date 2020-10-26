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

    Tensor* t1 = new Tensor::Tensor({true,false,false,false,true,true}, {6}, DEV_CPU);
    bool condition =  Tensor::all(t1);
    //condition = false
    

any
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::any

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({true,false,false,false,true,true}, {6}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({false,false,false,false,false,false}, {6}, DEV_CPU);

    bool condition =  Tensor::any(t1);
    //condition = true

    bool condition2 =  Tensor::any(t2);
    //condition2 = false


Array contents
-----------------



isfinite
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isfinite

.. code-block:: c++

    Tensor* t1 = Tensor::full({4}, 5.0f);
    Tensor* r1 = nullptr;

    Tensor::isfinite(t1, r1);
    //[true, true, true, true]

    

isinf
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isinf

.. code-block:: c++

    Tensor* t1 = Tensor::full({4}, 5.0f);
    Tensor* r1 = nullptr;

    Tensor::isinf(t1, r1);
    //[false, false, false, false]

isnan
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isnan

.. code-block:: c++

    Tensor* t1 = Tensor::full({4}, 5.0f);
    Tensor* r1 = nullptr;

    Tensor::isnan(t1, r1);
    //[false, false, false, false]
    

isneginf
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isneginf

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({-INFINITY, INFINITY, 1.0, 2.0}, {4}, DEV_CPU);
    Tensor* r1;


    Tensor::isneginf(t1, r1);
    // r1 => [true, false, false, false]
    

isposinf
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isposinf

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({-INFINITY, INFINITY, 1.0, 2.0}, {4}, DEV_CPU);
    Tensor* r1;


    Tensor::isposinf(t1, r1);
    // r1 => [false, true, false, false]



Logical operations
---------------------------


logical_and
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logical_and

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({true,false,true,false,true,true}, {6}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({false,false,true,false,false,false}, {6}, DEV_CPU);
    Tensor* r;

    Tensor::logical_and(t1, t2, r);
    // r => [false, false, true, false, false, false]
        

logical_or
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logical_or

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({true,false,true,false,true,true}, {6}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({false,false,true,false,false,false}, {6}, DEV_CPU);
    Tensor* r;

    Tensor::logical_or(t1, t2, r);
    // r => [true, false, true, false, true, true]
        

logical_not
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logical_not

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({true,false,true,false,true,true}, {6}, DEV_CPU);
    Tensor* r;

    Tensor::logical_and(t1, r);
    // r => [false, true, false, true, false, false]
        

logical_xor
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logical_xor

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({true,false,true,false,true,true}, {6}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({false,false,true,false,false,false}, {6}, DEV_CPU);
    Tensor* r;

    Tensor::logical_xor(t1, t2, r);
    // r => [true, false, false, false, true, true]



Comparison
---------------------------


allclose
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::allclose

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({10000.0, 1e-07}, {2}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({10000.0, 1e-08}, {2}, DEV_CPU);

    bool close =  Tensor::allclose(t1, t2, 1e-05, 1e-08, false);  
    // close = false


    Tensor* t1 = new Tensor::Tensor({10000.0, 1e-08}, {2}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({10000.0, 1e-09}, {2}, DEV_CPU);

    bool close =  Tensor::allclose(t1, t2, 1e-05, 1e-08, false);  
    // close = true

    Tensor* t1 = new Tensor::Tensor({1.0, NAN}, {2}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({1.0, NAN}, {2}, DEV_CPU);

    bool close =  Tensor::allclose(t1, t2, 1e-05, 1e-08, false);  
    // close = false

    Tensor* t1 = new Tensor::Tensor({1.0, NAN}, {2}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({1.0, NAN}, {2}, DEV_CPU);

    bool close =  Tensor::allclose(t1, t2, 1e-05, 1e-08, true);  
    // close = true
    

isclose
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isclose

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({10000.0, 1e-07}, {2}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({10000.0, 1e-08}, {2}, DEV_CPU);
    Tensor* r;

    Tensor::isclose(t1, t2, r, 1e-05, 1e-08, false);  
    // r => [true, false]


    Tensor* t1 = new Tensor::Tensor({10000.0, 1e-08}, {2}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({10000.0, 1e-09}, {2}, DEV_CPU);
    Tensor* r;

    Tensor::isclose(t1, t2, r, 1e-05, 1e-08, false);  
    // r => [true, true]

    Tensor* t1 = new Tensor::Tensor({1.0, NAN}, {2}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({1.0, NAN}, {2}, DEV_CPU);
    Tensor* r;

    Tensor::isclose(t1, t2, r, 1e-05, 1e-08, false);  
    // r => [true, false]

    Tensor* t1 = new Tensor::Tensor({1.0, NAN}, {2}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({1.0, NAN}, {2}, DEV_CPU);
    Tensor* r;

    Tensor::isclose(t1, t2, r, 1e-05, 1e-08, false);  
    // r => [true, true]

        

greater
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::greater_(float)
.. doxygenfunction:: Tensor::greater(float v)
.. doxygenfunction:: Tensor::greater(Tensor *A, Tensor *B, float v)
.. doxygenfunction:: Tensor::greater(Tensor *A)
.. doxygenfunction:: Tensor::greater(Tensor *A, Tensor *B, Tensor *C)

 
.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({10000.0, 1e-07}, {2}, DEV_CPU);
    Tensor* r;

    r = t1->greater(900.0);
    // r => [true, false]


    Tensor::greater(t1, r, 900.0);
    // r => [true, false]

    Tensor* t2 = new Tensor::Tensor({900.0, 1e-08}, {2}, DEV_CPU);
    Tensor* r2 =  t1->greater(t2);
    // r2 => [true, true]


    Tensor::greater(t1, t2, r);
    // r => [true, true]


    t1->greater_(900.0);
    // t1 => [true, false]


greater_equal
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::greater_equal_(float)
.. doxygenfunction:: Tensor::greater_equal(float v)
.. doxygenfunction:: Tensor::greater_equal(Tensor *A, Tensor *B, float v)
.. doxygenfunction:: Tensor::greater_equal(Tensor *A)
.. doxygenfunction:: Tensor::greater_equal(Tensor *A, Tensor *B, Tensor *C)


.. code-block:: c++


    Tensor* t1 = new Tensor::Tensor({10000.0, 1e-07}, {2}, DEV_CPU);
    Tensor* r;

    r = t1->greater_equal(10000.0);
    // r => [true, false]


    Tensor::greater_equal(t1, r, 10000.0);
    // r => [true, false]

    Tensor* t2 = new Tensor::Tensor({10000.0, 1e-08}, {2}, DEV_CPU);
    Tensor* r2 =  t1->greater_equal(t2);
    // r2 => [true, true]


    Tensor::greater_equal(t1, t2, r);
    // r => [true, true]


    t1->greater_equal_(10000.0);
    // t1 => [true, false]




less
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::less_(float)
.. doxygenfunction:: Tensor::less(float v)
.. doxygenfunction:: Tensor::less(Tensor *A, Tensor *B, float v)
.. doxygenfunction:: Tensor::less(Tensor *A)
.. doxygenfunction:: Tensor::less(Tensor *A, Tensor *B, Tensor *C)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({10000.0, 1e-07}, {2}, DEV_CPU);
    Tensor* r;

    r = t1->less(20000.0);
    // r => [true, true]


    Tensor::less(t1, r, 20000.0);
    // r => [true, true]

    Tensor* t2 = new Tensor::Tensor({20000.0, 1e-05}, {2}, DEV_CPU);
    Tensor* r2 =  t1->less(t2);
    // r2 => [true, true]


    Tensor::less(t1, t2, r);
    // r => [true, true]


    t1->less_(20000.0);
    // t1 => [true, true]



less_equal
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::less_equal_(float)
.. doxygenfunction:: Tensor::less_equal(float v)
.. doxygenfunction:: Tensor::less_equal(Tensor *A, Tensor *B, float v)
.. doxygenfunction:: Tensor::less_equal(Tensor *A)
.. doxygenfunction:: Tensor::less_equal(Tensor *A, Tensor *B, Tensor *C)


.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({10000.0, 1e-07}, {2}, DEV_CPU);
    Tensor* r;

    r = t1->less_equal(10000.0);
    // r => [true, true]


    Tensor::less_equal(t1, r, 10000.0);
    // r => [true, true]

    Tensor* t2 = new Tensor::Tensor({10000.0, 1e-05}, {2}, DEV_CPU);
    Tensor* r2 =  t1->less_equal(t2);
    // r2 => [true, true]


    Tensor::less_equal(t1, t2, r);
    // r => [true, true]


    t1->less_equal_(10000.0);
    // t1 => [true, true]
    


equal
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::equal_(float)
.. doxygenfunction:: Tensor::equal(float v)
.. doxygenfunction:: Tensor::equal(Tensor *A, Tensor *B, float v)
.. doxygenfunction:: Tensor::equal(Tensor *A)
.. doxygenfunction:: Tensor::equal(Tensor *A, Tensor *B, Tensor *C)


.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({10000.0, 1e-07}, {2}, DEV_CPU);
    Tensor* r;

    r = t1->equal(10000.0);
    // r => [true, false]


    Tensor::equal(t1, r, 10000.0);
    // r => [true, false]

    Tensor* t2 = new Tensor::Tensor({10000.0, 1e-05}, {2}, DEV_CPU);
    Tensor* r2 =  t1->equal(t2);
    // r2 => [true, false]


    Tensor::equal(t1, t2, r);
    // r => [true, false]


    t1->equal_(10000.0);
    // t1 => [true, false]


    
        

not_equal
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::not_equal_(float)
.. doxygenfunction:: Tensor::not_equal(float v)
.. doxygenfunction:: Tensor::not_equal(Tensor *A, Tensor *B, float v)
.. doxygenfunction:: Tensor::not_equal(Tensor *A)
.. doxygenfunction:: Tensor::not_equal(Tensor *A, Tensor *B, Tensor *C)



.. code-block:: c++


    Tensor* t1 = new Tensor::Tensor({10000.0, 1e-07}, {2}, DEV_CPU);
    Tensor* r;

    r = t1->not_equal(10000.0);
    // r => [false, true]


    Tensor::not_equal(t1, r, 10000.0);
    // r => [false, true]

    Tensor* t2 = new Tensor::Tensor({10000.0, 1e-05}, {2}, DEV_CPU);
    Tensor* r2 =  t1->not_equal(t2);
    // r2 => [false, true]


    Tensor::not_equal(t1, t2, r);
    // r => [false, true]


    t1->not_equal_(10000.0);
    // t1 => [false, true]

