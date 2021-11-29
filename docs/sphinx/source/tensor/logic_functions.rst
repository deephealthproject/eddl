Logic functions
===============

.. note::
   A practical example of the vast majority of the operations in this page are included in a working example in our `GitHub respository <https://github.com/deephealthproject/eddl/blob/master/examples/tensor/eddl_ops.cpp>`_

Truth value testing
---------------------------


All
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::all()

.. code-block:: c++

    Tensor* t1 = new Tensor({true,false,false,false,true,true}, {2, 3});
    bool condition =  t1->all(); //returns new tensor
    //condition = false

    //Other ways
    bool condition =  Tensor::all(t1); //source
    

Any
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::any()

.. code-block:: c++

    Tensor* t1 = new Tensor({true,false,false,false,true,true}, {2, 3});
    
    bool condition =  t1->any(); //returns new tensor
    //condition = true

    //Other ways
    bool condition =  Tensor::any(t1); //source


Array contents
-----------------


Is finite?
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isfinite()

.. code-block:: c++

    Tensor* t1 = new Tensor({12, INFINITY, NAN, -INFINITY, 0.0f, +INFINITY}, {2,3});
    // [
    // [12.00 inf nan]
    // [-inf 0.00 inf]
    // ]

    Tensor* r1 = t1->isfinite(); // returns new tensor
    
    r1->print(2);  // Temp.
    // [
    // [1.00 0.00 0.00]
    // [0.00 1.00 0.00]
    // ]

    //Other ways
    Tensor::isfinite(t1, r1); // static
    

Is inf?
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isinf()

.. code-block:: c++

    Tensor* t1 = new Tensor({12, INFINITY, NAN, -INFINITY, 0.0f, +INFINITY}, {2,3});
    // [
    // [12.00 inf nan]
    // [-inf 0.00 inf]
    // ]

    Tensor* r1 = t1->isinf(); // returns new tensor
    // [
    // [0.00 1.00 0.00]
    // [1.00 0.00 1.00]
    // ]

    //Other ways
    Tensor::isinf(t1, r1); // static

Is NaN?
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isnan()

.. code-block:: c++

    Tensor* t1 = new Tensor({12, INFINITY, NAN, -INFINITY, 0.0f, +INFINITY}, {2,3});
    // [
    // [12.00 inf nan]
    // [-inf 0.00 inf]
    // ]

    Tensor* r1 = t1->isnan(); // returns new tensor
    // [
    // [0.00 0.00 1.00]
    // [0.00 0.00 0.00]
    // ]
    
    //Other ways
    Tensor::isnan(t1, r1); // static


Is -inf?
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isneginf()

.. code-block:: c++

    Tensor* t1 = new Tensor({12, INFINITY, NAN, -INFINITY, 0.0f, +INFINITY}, {2,3});
    // [
    // [12.00 inf nan]
    // [-inf 0.00 inf]
    // ]

    Tensor* r1 = t1->isneginf(); // returns new tensor
    // [
    // [0.00 0.00 0.00]
    // [1.00 0.00 0.00]
    // ]

    //Other ways
    Tensor::isneginf(t1, r1); // static

    

Is +inf?
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isposinf()

.. code-block:: c++

    Tensor* t1 = new Tensor({12, INFINITY, NAN, -INFINITY, 0.0f, +INFINITY}, {2,3});
    // [
    // [12.00 inf nan]
    // [-inf 0.00 inf]
    // ]

    Tensor* r1 = t1->isposinf(); // returns new tensor
    // [
    // [0.00 1.00 0.00]
    // [0.00 0.00 1.00]
    // ]

    //Other ways
    Tensor::isposinf(t1, r1); // static




Logical operations
---------------------------


Logical AND: "A & B"
^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logical_and(Tensor *A)

.. code-block:: c++

    Tensor* t1 = Tensor::full({5,5}, 1.0f);
    // [
    // [1.00 1.00 0.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]


    Tensor* t2 = Tensor::full({5,5}, 0.0f);
    // [
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // ]

    Tensor* r = t1->logical_and(t2); // returns new tensor
    // [
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // ]

    //Other ways
    Tensor::logical_and(t1, t2, r); // static
        

Logical OR: "A | B"
^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logical_or(Tensor *A)

.. code-block:: c++

    Tensor* t1 = Tensor::full({5,5}, 1.0f);
    // [
    // [1.00 1.00 0.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]


    Tensor* t2 = Tensor::full({5,5}, 0.0f);
    // [
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // ]

    Tensor* r = t1->logical_or(t2); // returns new tensor
    // [
    // [1.00 1.00 0.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]

    //Other ways
    Tensor::logical_or(t1, t2, r); // static

        

Logical NOT: "~A"
^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logical_not()

.. code-block:: c++

    Tensor* t1 = Tensor::full({5,5}, 1.0f);
    // [
    // [1.00 1.00 0.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]

    Tensor* r = t1->logical_not(); // returns new tensor
    // [
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // ]

    //Other ways
    Tensor::logical_not(t1, r); // static

        

Logical XOR (Exclusive OR): "A ^ B"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logical_xor(Tensor *A)

.. code-block:: c++

    Tensor* t1 = Tensor::full({5,5}, 1.0f);
    // [
    // [1.00 1.00 0.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]


    Tensor* t2 = Tensor::full({5,5}, 0.0f);
    // [
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // ]

    Tensor* r = t1->logical_xor(t2); // returns new tensor
    // [
    // [1.00 1.00 0.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]

    //Other ways
    Tensor::logical_xor(t1, t2, r); // static




Comparison
---------------------------

Unary Operations
^^^^^^^^^^^^^^^^^^^^

Greater than: "A > B"
^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::greater(float v)

 
.. code-block:: c++

    Tensor* t1 = Tensor::range(1.0f, 25.0f, 1); t1->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    t1->greater_(3.0f); // In-place
    // [
    // [0.00 0.00 0.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]

    // Other Ways
    Tensor* t2 = t1->greater(3.0f); // returns new tensor
    Tensor::greater(t1, t2, 3.0f); // static


Greater equal: "A >= B"
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::greater_equal(float v)


.. code-block:: c++


    Tensor* t1 = Tensor::range(1.0f, 25.0f, 1); t1->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    t1->greater_equal_(3.0f); // In-place
    // [
    // [0.00 0.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]

    // Other Ways
    Tensor* t2 = t1->greater_equal(3.0f); // returns new tensor
    Tensor::greater_equal(t1, t2, 3.0f); // static




Less than: "A < B"
^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::less(float v)

.. code-block:: c++

    Tensor* t1 = Tensor::range(1.0f, 25.0f, 1); t1->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    t1->less_(3.0f); // In-place
    // [
    // [1.00 1.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // ]

    // Other Ways
    Tensor* t2 = t1->less(3.0f); // returns new tensor
    Tensor::less_(t1, t2, 3.0f); // static



Less equal: "A <= B"
^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::less_equal(float v)


.. code-block:: c++

    Tensor* t1 = Tensor::range(-2, 3); t1->reshape_({2, 3});

    Tensor* t2 = Tensor::randn({2, 3});

    Tensor* t3 = t1->less_equal(t2); // returns new tensor

    // Other Ways
    Tensor::less_equal(t1, t2, t3); // static
    


Equal: "A == B"
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::equal(float v)


.. code-block:: c++

    Tensor* t1 = Tensor::range(1.0f, 25.0f, 1); t1->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    t1->equal_(3.0f); // In-place
    // [
    // [0.00 0.00 1.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // ]

    // Other Ways
    Tensor* t2 = t1->equal(3.0f); // returns new tensor
    Tensor::equal(t1, t2, 3.0f); // static


    
        

Not Equal: "A != B"
^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::not_equal(float v)



.. code-block:: c++


    Tensor* t1 = Tensor::range(1.0f, 25.0f, 1); t1->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    t1->not_equal_(3.0f); // In-place
    // [
    // [1.00 1.00 0.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]

    // Other Ways
    Tensor* t2 = t1->not_equal(3.0f); // returns new tensor
    Tensor::not_equal(t1, t2, 3.0f); // static



Binary Operations
---------------------------


All Close?
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::allclose(Tensor *A, float rtol = 1e-05, float atol = 1e-08, bool equal_nan = false)

.. code-block:: c++

    Tensor* t1 = Tensor::range(1.0f, 25.0f); t1->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]


    Tensor* t2 = Tensor::range(1.0f, 25.0f); t2->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    float result = t1->allclose(t2);
    // 1.00

    //Other ways
    result = Tensor::allclose(t1, t2); //static
    

Is Close?
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isclose(Tensor *A, float rtol = 1e-05, float atol = 1e-08, bool equal_nan = false)

.. code-block:: c++

    Tensor* t1 = Tensor::range(1.0f, 25.0f); t1->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]


    Tensor* t2 = Tensor::range(1.0f, 25.0f); t2->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    Tensor* t3 = t1->isclose(t2); // returns new tensor
    // [
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]

    //Other ways
    Tensor::isclose(t1, t2, t3); //static

        

Greater Than: "A > B"
^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::greater(Tensor *A)

 
.. code-block:: c++

    Tensor* t1 = Tensor::range(1.0f, 25.0f); t1->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]


    Tensor* t2 = Tensor::range(1.0f, 25.0f); t2->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    Tensor* t3 = t1->greater(t2); // returns new tensor
    // [
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // ]

    // Other Ways
    Tensor::greaterl(t1, t2, t3); // static


Greater Equal: "A >= B"
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::greater_equal(Tensor *A)


.. code-block:: c++


    Tensor* t1 = Tensor::range(1.0f, 25.0f); t1->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]


    Tensor* t2 = Tensor::range(1.0f, 25.0f); t2->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    Tensor* t3 = t1->greater_equal(t2); // returns new tensor
    // [
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]

    // Other Ways
    Tensor::greater_equal(t1, t2, t3); // static




Less Than: "A < B"
^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::less(Tensor *A)

.. code-block:: c++

    
    Tensor* t1 = Tensor::range(1.0f, 25.0f); t1->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]


    Tensor* t2 = Tensor::range(1.0f, 25.0f); t2->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    Tensor* t3 = t1->less(t2); // returns new tensor
    // [
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // ]

    // Other Ways
    Tensor::less(t1, t2, t3); // static



Less Equal: "A <= B"
^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::less_equal(Tensor *A)


.. code-block:: c++

    Tensor* t1 = Tensor::range(1.0f, 25.0f); t1->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]


    Tensor* t2 = Tensor::range(1.0f, 25.0f); t2->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    Tensor* t3 = t1->less_equal(t2); // returns new tensor
    // [
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]

    // Other Ways
    Tensor::less_equal(t1, t2, t3); // static


Equal: "A == B"
^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::equal(Tensor *A)


.. code-block:: c++
    
    Tensor* t1 = Tensor::range(1.0f, 25.0f); t1->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]


    Tensor* t2 = Tensor::range(1.0f, 25.0f); t2->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    Tensor* t3 = t1->equal(t2); // returns new tensor
    // [
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]

    // Other Ways
    Tensor::equal(t1, t2, t3); // static


    
        

Not Equal: "A != B"
^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::not_equal(Tensor *A)


.. code-block:: c++


    Tensor* t1 = Tensor::range(1.0f, 25.0f); t1->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]


    Tensor* t2 = Tensor::range(1.0f, 25.0f); t2->reshape_({5,5});
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    Tensor* t3 = t1->not_equal(t2); // returns new tensor
    // [
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // ]

    // Other Ways
    Tensor::not_equal(t1, t2, t3); // static

