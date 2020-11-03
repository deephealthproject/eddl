Logic functions
===============

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/docs/markdown/eddl_progress_tensor.md


Truth value testing
---------------------------


All
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::all

.. code-block:: c++

    Tensor* t1 = new Tensor({true,false,false,false,true,true}, {2, 3});
    t1->print(2);  // Temp.
    bool condition =  Tensor::all(t1);
    cout << condition << endl; // Temp.
    //condition = false
    

Any
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::any

.. code-block:: c++

    Tensor* t1 = new Tensor({true,false,false,false,true,true}, {2, 3});
    t1->print(2);  // Temp.

    
    bool condition =  Tensor::any(t1);
    cout << condition << endl; // Temp.
    //condition = true


Array contents
-----------------


Is finite?
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isfinite

.. code-block:: c++

    Tensor* t1 = new Tensor({12, INFINITY, NAN, -INFINITY, 0.0f, +INFINITY}, {2,3});
    t1->print(2);  // Temp.
    // [
    // [12.00 inf nan]
    // [-inf 0.00 inf]
    // ]

    Tensor* r1 = nullptr;
    Tensor::isneginf(t1, r1); // static
    r1->print(2);  // Temp.
    // [
    // [1.00 0.00 0.00]
    // [0.00 1.00 0.00]
    // ]

    

Is inf?
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isinf

.. code-block:: c++

    Tensor* t1 = new Tensor({12, INFINITY, NAN, -INFINITY, 0.0f, +INFINITY}, {2,3});
    t1->print(2);  // Temp.
    // [
    // [12.00 inf nan]
    // [-inf 0.00 inf]
    // ]

    Tensor* r1 = nullptr;
    Tensor::isneginf(t1, r1); // static
    r1->print(2);  // Temp.
    // [
    // [0.00 1.00 0.00]
    // [1.00 0.00 1.00]
    // ]

Is NaN?
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isnan

.. code-block:: c++

    Tensor* t1 = new Tensor({12, INFINITY, NAN, -INFINITY, 0.0f, +INFINITY}, {2,3});
    t1->print(2);  // Temp.
    // [
    // [12.00 inf nan]
    // [-inf 0.00 inf]
    // ]

    Tensor* r1 = nullptr;
    Tensor::isneginf(t1, r1); // static
    r1->print(2);  // Temp.
    // [
    // [0.00 0.00 1.00]
    // [0.00 0.00 0.00]
    // ]
    

Is -inf?
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isneginf

.. code-block:: c++

    Tensor* t1 = new Tensor({12, INFINITY, NAN, -INFINITY, 0.0f, +INFINITY}, {2,3});
    t1->print(2);  // Temp.
    // [
    // [12.00 inf nan]
    // [-inf 0.00 inf]
    // ]

    Tensor* r1 = nullptr;
    Tensor::isneginf(t1, r1); // static
    r1->print(2);  // Temp.
    // [
    // [0.00 0.00 0.00]
    // [1.00 0.00 0.00]
    // ]
    

Is +inf?
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isposinf

.. code-block:: c++

    Tensor* t1 = new Tensor({12, INFINITY, NAN, -INFINITY, 0.0f, +INFINITY}, {2,3});
    t1->print(2);  // Temp.
    // [
    // [12.00 inf nan]
    // [-inf 0.00 inf]
    // ]

    Tensor* r1 = nullptr;
    Tensor::isneginf(t1, r1); // static
    r1->print(2);  // Temp.
    // [
    // [0.00 1.00 0.00]
    // [0.00 0.00 1.00]
    // ]



Logical operations
---------------------------


Logical AND: "A & B"
^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logical_and

.. code-block:: c++

    Tensor* t1 = Tensor::full({5,5}, 1.0f);
    t1->print(2);  // Temp.
    // [
    // [1.00 1.00 0.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]


    Tensor* t2 = Tensor::full({5,5}, 0.0f);
    t2->print(2);  // Temp.
    // [
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // ]

    Tensor* r = nullptr;

    Tensor::logical_and(t1, t2, r); // static
    r->print(2);  // Temp.
    // [
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // ]
        

Logical OR: "A | B"
^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logical_or

.. code-block:: c++

    Tensor* t1 = Tensor::full({5,5}, 1.0f);
    t1->print(2);  // Temp.
    // [
    // [1.00 1.00 0.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]


    Tensor* t2 = Tensor::full({5,5}, 0.0f);
    t2->print(2);  // Temp.
    // [
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // ]

    Tensor* r = nullptr;

    Tensor::logical_or(t1, t2, r); // static
    r->print(2);  // Temp.
    // [
    // [1.00 1.00 0.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]
        

Logical NOT: "~A"
^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logical_not

.. code-block:: c++

    Tensor* t1 = Tensor::full({5,5}, 1.0f);
    t1->print(2);  // Temp.
    // [
    // [1.00 1.00 0.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]

    Tensor* r = nullptr;

    Tensor::logical_not(t1, r); // static
    r->print(2);  // Temp.
    // [
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // ]
        

Logical XOR (Exclusive OR): "A ^ B"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logical_xor

.. code-block:: c++

    Tensor* t1 = Tensor::full({5,5}, 1.0f);
    t1->print(2);  // Temp.
    // [
    // [1.00 1.00 0.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]


    Tensor* t2 = Tensor::full({5,5}, 0.0f);
    t2->print(2);  // Temp.
    // [
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // ]

    Tensor* r = nullptr;

    Tensor::logical_xor(t1, t2, r); // static
    r->print(2);  // Temp.
    // [
    // [1.00 1.00 0.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]



Comparison
---------------------------

Unary Operations
^^^^^^^^^^^^^^^^^^^^

Greater than: "A > B"
^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::greater(float v)

 
.. code-block:: c++

    Tensor* t1 = Tensor::range(1.0f, 25.0f, 1); t1->reshape_({2,3});
    t1->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    t1->greater_(3.0f); // In-place
    t1->print(2);  // Temp.
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


    Tensor* t1 = Tensor::range(1.0f, 25.0f, 1); t1->reshape_({2,3});
    t1->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    t1->greater_equal_(3.0f); // In-place
    t1->print(2);  // Temp.
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

    Tensor* t1 = Tensor::range(1.0f, 25.0f, 1); t1->reshape_({2,3});
    t1->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    t1->less_(3.0f); // In-place
    t1->print(2);  // Temp.
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
    t1->print(2);  // Temp.

    Tensor* t2 = Tensor::randn({2, 3});
    t2->print(2);  // Temp.

    Tensor* t3 = t1->less_equal(t2); // returns new tensor
    t3->print(2);  // Temp.

    // Other Ways
    Tensor::less_equal(t1, t2, t3); // static
    


Equal: "A == B"
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::equal(float v)


.. code-block:: c++

    Tensor* t1 = Tensor::range(1.0f, 25.0f, 1); t1->reshape_({2,3});
    t1->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    t1->equal_(3.0f); // In-place
    t1->print(2);  // Temp.
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


    Tensor* t1 = Tensor::range(1.0f, 25.0f, 1); t1->reshape_({2,3});
    t1->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    t1->not_equal_(3.0f); // In-place
    t1->print(2);  // Temp.
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
^^^^^^^^^^^^^^^^^^^^^


All Close?
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::allclose

.. code-block:: c++

    Tensor* t1 = Tensor::range(1, 6); t1->reshape_({2,3});
    t1->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]


    Tensor* t2 = Tensor::range(1, 6); t2->reshape_({2,3});
    t2->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    float result = Tensor::allclose(t1, t2);
    cout << result << endl;
    // 1.00
    

Is Close?
^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isclose

.. code-block:: c++

    Tensor* t1 = Tensor::range(1, 6); t1->reshape_({2,3});
    t1->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]


    Tensor* t2 = Tensor::range(1, 6); t2->reshape_({2,3});
    t2->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    Tensor* t3 = new Tensor({5,5});


    Tensor::isclose(t1, t2, t3);
    t3->print(2);  // Temp.  
    // [
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // [1.00 1.00 1.00 1.00 1.00]
    // ]

        

Greater Than: "A > B"
^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::greater(Tensor *A)

 
.. code-block:: c++

    Tensor* t1 = Tensor::range(1, 6); t1->reshape_({2,3});
    t1->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]


    Tensor* t2 = Tensor::range(1, 6); t2->reshape_({2,3});
    t2->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    Tensor* t3 = t1->greater(t2); // returns new tensor
    t3->print(2);  // Temp.
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


    Tensor* t1 = Tensor::range(1, 6); t1->reshape_({2,3});
    t1->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]


    Tensor* t2 = Tensor::range(1, 6); t2->reshape_({2,3});
    t2->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    Tensor* t3 = t1->greater_equal(t2); // returns new tensor
    t3->print(2);  // Temp.
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

    
    Tensor* t1 = Tensor::range(1, 6); t1->reshape_({2,3});
    t1->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]


    Tensor* t2 = Tensor::range(1, 6); t2->reshape_({2,3});
    t2->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    Tensor* t3 = t1->less(t2); // returns new tensor
    t3->print(2);  // Temp.
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

    Tensor* t1 = Tensor::range(1, 6); t1->reshape_({2,3});
    t1->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]


    Tensor* t2 = Tensor::range(1, 6); t2->reshape_({2,3});
    t2->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    Tensor* t3 = t1->less_equal(t2); // returns new tensor
    t3->print(2);  // Temp.
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
    
    Tensor* t1 = Tensor::range(1, 6); t1->reshape_({2,3});
    t1->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]


    Tensor* t2 = Tensor::range(1, 6); t2->reshape_({2,3});
    t2->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    Tensor* t3 = t1->equal(t2); // returns new tensor
    t3->print(2);  // Temp.
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


    Tensor* t1 = Tensor::range(1, 6); t1->reshape_({2,3});
    t1->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]


    Tensor* t2 = Tensor::range(1, 6); t2->reshape_({2,3});
    t2->print(2);  // Temp.
    // [
    // [1.00 2.00 3.00 4.00 5.00]
    // [6.00 7.00 8.00 9.00 10.00]
    // [11.00 12.00 13.00 14.00 15.00]
    // [16.00 17.00 18.00 19.00 20.00]
    // [21.00 22.00 23.00 24.00 25.00]
    // ]

    Tensor* t3 = t1->not_equal(t2); // returns new tensor
    t3->print(2);  // Temp.
    // [
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00 0.00]
    // ]

    // Other Ways
    Tensor::not_equal(t1, t2, t3); // static

