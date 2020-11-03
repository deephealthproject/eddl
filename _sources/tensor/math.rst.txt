Mathematical functions
========================

Point-wise
-----------

Abs
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::abs()

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2, 3});
    // [
    // [0.31 0.08 1.59]
    // [-0.77 -0.46 -0.39]
    // ]

    t1->abs_();  // In-place
    // [
    // [0.31 0.08 1.59]
    // [0.77 0.46 0.39]
    // ]

    // Other ways
    Tensor* t2 = t1->abs(); // returns a new tensor
    Tensor::abs(t1, t2); // static


Acos
^^^^^^^^^^^^
.. doxygenfunction:: Tensor::acos()


.. code-block:: c++

    Tensor* t1 = Tensor::randn({2, 3});
    // [
    // [0.42 -0.80 0.27]
    // [0.42 -0.80 0.27]
    // ]

    t1->acos_(); // In-place
    // [
    // [1.13 2.51 1.29]
    // [1.12 0.78 1.77]
    // ]

    // Other ways
    Tensor* t2 = t1->acos(); // returns a new tensor
    Tensor::acos(t1, t2); // static
    
    
Add
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::add(float v)

.. code-block:: c++

    Tensor* t1 = Tensor::full({2, 3}, 5.0f);
    // [
    // [5.00 5.00 5.00]
    // [5.00 5.00 5.00]
    // ]

    t1->add_(2.0f); // In-place
    // [
    // [7.00 7.00 7.00]
    // [7.00 7.00 7.00]
    // ]

    // Other ways
    Tensor* t2 = t1->add(2.0f); // returns new tensor
    Tensor::add(t1, t2, 2.0f); // static


Asin
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::asin()

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2, 3});
    // [
    // [-0.99 0.24 0.39]
    // [-0.01 1.64 0.01]
    // ]

    t1->asin_(); // In-place
    // [
    // [-1.54 0.24 0.40]
    // [-0.01 nan 0.01]
    // ]

    // Other ways
    Tensor* t2 = t1->asin(); // returns a new tensor
    Tensor::asin(t1, t2); // static

    
Atan
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::atan()

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2, 3});
    // [
    // [-0.82 -0.04 0.29]
    // [-0.80 -0.03 0.51]
    // ]

    t1->atan_(); // In-place
    // [
    // [-0.68 -0.04 0.29]
    // [-0.67 -0.03 0.47]
    // ]

    // Other ways
    Tensor* t2 = t1->atan(); // returns a new tensor
    Tensor::atan(t1, t2); // static


Ceil
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::ceil()

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2, 3});
    // [
    // [-0.15 0.63 -0.76]
    // [0.18 -0.12 0.18]
    // ]

    t1->ceil_(); // In-place
    // [
    // [-0.00 1.00 -0.00]
    // [1.00 -0.00 1.00]
    // ]

    // Other ways
    Tensor* t2 = t1->ceil(); // returns a new tensor
    Tensor::ceil(t1, t2); // static


Clamp
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::clamp(float min, float max)

.. code-block:: c++

    Tensor* t1 = Tensor::range(-5.0f, 5.0f);
    // [-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00]

    t1->clamp_(-2.0f, 3.0f); // In-place
    // [-2.00 -2.00 -2.00 -2.00 -1.00 0.00 1.00 2.00 3.00 3.00 3.00]

    // Other ways
    Tensor* t2 = t1->clamp(-2.0f, 3.0f); // returns a new tensor
    Tensor::clamp(t1, t2, -2.0f, 3.0f); // static


Clampmax
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::clampmax(float max)

.. code-block:: c++

    Tensor* t1 = Tensor::range(-5.0f, 5.0f);
    // [-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00]

    t1->clampmax_(3.0f); // In-place
    // [-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 3.00 3.00]

    // Other ways
    Tensor* t2 = t1->clampmax(3.0f); // returns a new tensor
    Tensor::clampmax(t1, t2, 3.0f); // static
   

    
Clampmin
^^^^^^^^^^^^


.. doxygenfunction:: Tensor::clampmin(float min)

.. code-block:: c++

     Tensor* t1 = Tensor::range(-5.0f, 5.0f);
    // [-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00]

    t1->clampmin_(3.0f); // In-place
    // [-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 3.00 3.00]

    // Other ways
    Tensor* t2 = t1->clampmin(3.0f); // returns a new tensor
    Tensor::clampmin(t1, t2, 3.0f); // static

    
Cos
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cos()

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2, 3});
    // [
    // [-0.58 0.45 -1.14]
    // [-0.24 -1.15 -1.33]
    // ]

    t1->cos_(); // In-place
    // [
    // [0.83 0.90 0.41]
    // [0.97 0.41 0.23]
    // ]

    // Other ways
    Tensor* t2 = t1->cos(); // returns a new tensor
    Tensor::cos(t1, t2); // static

    
Cosh
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cosh()

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2, 3});
    // [
    // [-1.52 -0.52 0.31]
    // [0.85 1.06 0.26]
    // ]

    t1->cosh_(); // In-place
    // [
    // [2.40 1.14 1.05]
    // [1.39 1.62 1.04]
    // ]

    // Other ways
    Tensor* t2 = t1->cosh(); // returns a new tensor
    Tensor::cosh(t1, t2); // static
  
    
Div
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::div(float v)

.. code-block:: c++

    Tensor* t1 = Tensor::full({2, 3}, 5.0f);
    // [
    // [5.00 5.00 5.00]
    // [5.00 5.00 5.00]
    // ]

    t1->div_(2.0f); // In-place
    // [
    // [2.50 2.50 2.50]
    // [2.50 2.50 2.50]
    // ]

    // Other ways
    Tensor* t2 = t1->div(2.0f); // returns new tensor
    Tensor::div(t1, t2, 2.0f); // static
    


Exp
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::exp()

.. code-block:: c++

    Tensor* t1 = Tensor::range({-5.0f, 5.0f);
    // [-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00]

    t1->exp_(); // In-place
    // [0.01 0.02 0.05 0.14 0.37 1.00 2.72 7.39 20.09 54.60 148.41]

    // Other ways
    Tensor* t2 = t1->exp(); // returns new tensor
    Tensor::exp(t1, t2); // static


Floor
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::floor()

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2, 3});
    // [
    // [0.47 1.39 0.94]
    // [0.98 1.16 0.40]
    // ]

    t1->floor_(); // In-place
    // [
    // [0.00 1.00 0.00]
    // [0.00 1.00 0.00]
    // ]

    // Other ways
    Tensor* t2 = t1->floor(); // returns new tensor
    Tensor::floor(t1, t2); // static


Inv
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::inv(float v = 1.0f)

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2, 3});
    // [
    // [0.58 -0.49 0.04]
    // [-1.25 -1.33 0.23]
    // ]

    t1->inv_(); // In-place
    // [
    // [1.72 -2.04 25.16]
    // [-0.80 -0.75 4.34]
    // ]

    // Other ways
    Tensor* t2 = t1->inv(); // returns new tensor
    Tensor::inv(t1, t2); // static


Log
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::log()

.. code-block:: c++

    Tensor* t1 = Tensor::range(-1.0f, 5.0f);
    // [-1.00 0.00 1.00 2.00 3.00 4.00 5.00]

    t1->log_(); // In-place
    // [-nan -inf 0.00 0.69 1.10 1.39 1.61]

    // Other ways
    Tensor* t2 = t1->log(); // returns new tensor
    Tensor::log(t1, t2); // static

    
Log2
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::log2()

.. code-block:: c++

    Tensor* t1 = Tensor::range(-1.0f, 5.0f);
    // [-1.00 0.00 1.00 2.00 3.00 4.00 5.00]

    t1->log2_(); // In-place
    // [-nan -inf 0.00 1.00 1.58 2.00 2.32]

    // Other ways
    Tensor* t2 = t1->log2(); // returns new tensor
    Tensor::log2(t1, t2); // static
  
    
Log10
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::log10()

.. code-block:: c++

    Tensor* t1 = Tensor::range(-1.0f, 5.0f);
    // [-1.00 0.00 1.00 2.00 3.00 4.00 5.00]

    t1->log10_(); // In-place
    // [nan -inf 0.00 0.30 0.48 0.60 0.70]

    // Other ways
    Tensor* t2 = t1->log10(); // returns new tensor
    Tensor::log10(t1, t2); // static
    
    
Logn
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logn(float n)

.. code-block:: c++

    Tensor* t1 = Tensor::range(-1.0f, 5.0f);
    // [-1.00 0.00 1.00 2.00 3.00 4.00 5.00]

    t1->logn_(5.0f); // In-place
    // [-nan -inf 0.00 0.43 0.68 0.86 1.00]

    // Other ways
    Tensor* t2 = t1->logn(5.0f); // returns new tensor
    Tensor::logn(t1, t2, 5.0f); // static


Maximum
^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::maximum(float v)

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2,3});
    // [
    // [-0.63 -0.10 0.01]
    // [-1.47 -0.84 0.54]
    // ]


    Tensor* t2 = t1->maximum(0.3f);  // returns a new tensor
    // [
    // [0.30 0.30 0.30]
    // [0.30 0.30 0.54]
    // ]

    // Other ways
    Tensor::maximum(t1, 0.3f) // source
    Tensor::maximum(t1, t2, 0.3f); // static



Minimum
^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::minimum(float v)

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2,3});
    // [
    // [-1.30 -1.05 0.49]
    // [0.03 0.09 0.28]
    // ]


    Tensor* t2 = t1->minimum(0.3f);  // returns a new tensor
    // [
    // [-1.30 -1.05 0.30]
    // [0.03 0.09 0.28]
    // ]

    // Other ways
    Tensor::minimum(t1, 0.3f) // source
    Tensor::minimum(t1, t2, 0.3f); // static


Mod
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::mod(float v)

.. code-block:: c++

    Tensor* t1 = Tensor::range(-5.0f, 5.0f);
    // [-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00]

    t1->mod_(2.0f); // In-place
    // [-1.00 -0.00 -1.00 -0.00 -1.00 0.00 1.00 0.00 1.00 0.00 1.00]

    // Other ways
    Tensor* t2 = t1->mod(2.0f); // returns new tensor
    Tensor::mod(t1, t2, 2.0f); // static

    
Mult
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::mult(float v)


.. code-block:: c++

    Tensor* t1 = Tensor::full({2, 3}, 5.0f);
    // [
    // [5.00 5.00 5.00]
    // [5.00 5.00 5.00]
    // ]

    t1->mult_(2.0); // In-place
    // [
    // [10.00 10.00 10.00]
    // [10.00 10.00 10.00]
    // ]

    // Other ways
    Tensor* t2 = t1->mult(2.0f); // returns new tensor
    Tensor::mult(t1, t2, 2.0f); // static
    
Neg
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::neg()

.. code-block:: c++

    Tensor* t1 = Tensor::full({2, 3}, 5.0f);
    // [
    // [5.00 5.00 5.00]
    // [5.00 5.00 5.00]
    // ]

    t1->neg_(); // In-place
    // [
    // [-5.00 -5.00 -5.00]
    // [-5.00 -5.00 -5.00]
    // ]

    // Other ways
    Tensor* t2 = t1->neg(); // returns new tensor
    Tensor::neg(t1, t2); // static


Normalize
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::normalize(float min = 0.0f, float max = 1.0f)

.. code-block:: c++

    Tensor* t1 = Tensor::range(-5.0f, 5.0f);
    // [-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00]

    t1->normalize_(0.0f, 1.0f); // In-place
    // [0.00 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00]

    // Other ways
    Tensor* t2 = t1->normalize(0.0f, 1.0f); // returns new tensor
    Tensor::normalize(t1, t2, 0.0f, 1.0f); // static
    
Pow
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::pow(float exp)

.. code-block:: c++

    Tensor* t1 = Tensor::full({2, 3}, 5.0f);
    // [
    // [5.00 5.00 5.00]
    // [5.00 5.00 5.00]
    // ]

    t1->pow_(2.0f); // In-place
    // [
    // [25.00 25.00 25.00]
    // [25.00 25.00 25.00]
    // ]

    // Other ways
    Tensor* t2 = t1->pow(2.0f); // returns new tensor
    Tensor::pow(t1, t2, 2.0f); // static


Powb
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::powb(float base)

.. code-block:: c++

    Tensor* t1 = Tensor::full({2, 3}, 5.0f);
    // [
    // [5.00 5.00 5.00]
    // [5.00 5.00 5.00]
    // ]

    t1->powb_(2.0f); // In-place
    // [
    // [32.00 32.00 32.00]
    // [32.00 32.00 32.00]
    // ]

    // Other ways
    Tensor* t2 = t1->powb(2.0f); // returns new tensor
    Tensor::powb(t1, t2, 2.0f); // static
    
Reciprocal
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::reciprocal()

.. code-block:: c++

    Tensor* t1 = Tensor::full({2, 3}, 5.0f);
    // [
    // [5.00 5.00 5.00]
    // [5.00 5.00 5.00]
    // ]

    t1->reciprocal_(2.0f); // In-place
    // [
    // [0.20 0.20 0.20]
    // [0.20 0.20 0.20]
    // ]

    // Other ways
    Tensor* t2 = t1->reciprocal(); // returns new tensor
    Tensor::reciprocal(t1, t2); // static
    
Remainder
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::remainder(float v)

.. code-block:: c++

    Tensor* t1 = Tensor::range(-5.0f, 5.0f);
    // [-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00]

    t1->remainder_(2.0f); // In-place
    // [-1.00 -0.00 -1.00 -0.00 -1.00 0.00 1.00 0.00 1.00 0.00 1.00]

    // Other ways
    Tensor* t2 = t1->remainder(2.0f); // returns new tensor
    Tensor::remainder(t1, t2, 2.0f); // static
    
    
Round
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::round()

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2, 3});
    // [
    // [1.14 -0.32 0.40]
    // [0.21 -0.38 0.01]
    // ]

    t1->round_(); // In-place
    // [
    // [1.00 -0.00 0.00]
    // [0.00 -0.00 0.00]
    // ]

    // Other ways
    Tensor* t2 = t1->round(); // returns new tensor
    Tensor::round(t1, t2); // static
    
Rsqrt
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::rsqrt()

.. code-block:: c++

    Tensor* t1 = Tensor::range(-1.0f, 5.0f);
    // [-1.00 0.00 1.00 2.00 3.00 4.00 5.00]

    t1->rsqrt_(); // In-place
    // [-nan inf 1.00 0.71 0.58 0.50 0.45]

    // Other ways
    Tensor* t2 = t1->rsqrt(); // returns new tensor
    Tensor::rsqrt(t1, t2); // static

Sigmoid
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sigmoid()

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2,3});
    // [
    // [0.11 0.87 0.18]
    // [2.13 -0.13 0.12]
    // ]


    t1->sigmoid_();  // In-place
    // [
    // [0.53 0.70 0.54]
    // [0.89 0.47 0.53]
    // ]

    // Other ways 
    Tensor* t2 = t1->sigmoid(); // returns a new tensor
    Tensor::sigmoid(t1, t2); // static
    
    
Sign
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sign(float zero_sign = 0.0f)


.. code-block:: c++

    Tensor* t1 = Tensor::linspace(-1.0f, 1.0f, 5.0f);
    // [-1.00 -0.50 0.00 0.50 1.00]


    t1->sign_(5.0f);  // In-place
    // [-1.00 -1.00 5.00 1.00 1.00]

    // Other ways
    Tensor* t2 = t1->sign(5.0f); // returns a new tensor
    Tensor::sign(t1, t2, 5.0f); // static
    

Sin
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sin()

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2,3});
    // [
    // [0.11 0.87 0.18]
    // [2.13 -0.13 0.12]
    // ]


    t1->sin_();  // In-place
    // [
    // [0.53 0.70 0.54]
    // [0.89 0.47 0.53]
    // ]

    // Other ways 
    Tensor* t2 = t1->sin(); // returns a new tensor
    Tensor::sin(t1, t2); // static

    
Sinh
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sinh()

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2,3});
    // [
    // [-0.35 -0.14 0.08]
    // [0.64 0.46 1.49]
    // ]


    t1->sinh_();  // In-place
    // [
    // [-0.36 -0.15 0.08]
    // [0.69 0.48 2.10]
    // ]

    // Other ways 
    Tensor* t2 = t1->sinh(); // returns a new tensor
    Tensor::sinh(t1, t2); // static
    
Sqr
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sqr()

.. code-block:: c++
    

    Tensor* t1 = Tensor::range(-1.0f, 5.0f);
    // [-1.00 0.00 1.00 2.00 3.00 4.00 5.00]

    t1->sqr_(); // In-place
    // [1.00 0.00 1.00 4.00 9.00 16.00 25.00]

    // Other ways
    Tensor* t2 = t1->sqr(); // returns new tensor
    Tensor::sqr(t1, t2); // static
    
Sqrt
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sqrt()

.. code-block:: c++

    Tensor* t1 = Tensor::range(-1.0f, 5.0f);
    // [-1.00 0.00 1.00 2.00 3.00 4.00 5.00]

    t1->sqrt_(); // In-place
    // [-nan 0.00 1.00 1.41 1.73 2.00 2.24]

    // Other ways
    Tensor* t2 = t1->sqrt(); // returns new tensor
    Tensor::sqrt(t1, t2); // static
    
Sub
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sub(float v)

.. code-block:: c++


    Tensor* t1 = Tensor::full({2, 3}, 5.0f);
    // [
    // [5.00 5.00 5.00]
    // [5.00 5.00 5.00]
    // ]

    t1->sub_(2.0f); // In-place
    // [
    // [3.00 3.00 3.00]
    // [3.00 3.00 3.00]
    // ]

    // Other ways
    Tensor* t2 = t1->sub(2.0f); // returns new tensor
    Tensor::sub(t1, t2, 2.0f); // static
    

    
Tan
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::tan()

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2,3});
    // [
    // [0.20 0.97 1.08]
    // [0.12 1.06 -0.67]
    // ]


    t1->tan_();  // In-place
    // [
    // [0.20 1.45 1.85]
    // [0.12 1.78 -0.80]
    // ]

    // Other ways 
    Tensor* t2 = t1->tan(); // returns a new tensor
    Tensor::tan(t1, t2); // static
    
Tanh
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::tanh()

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2,3});
    // [
    // [-2.07 -0.41 -0.83]
    // [0.12 0.45 0.90]
    // ]


    t1->tanh_();  // In-place
    // [
    // [-0.97 -0.39 -0.68]
    // [0.12 0.42 0.72]
    // ]

    // Other ways 
    Tensor* t2 = t1->tanh(); // returns a new tensor
    Tensor::tanh(t1, t2); // static
    
Trunc
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::trunc()

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2,3});
    // [
    // [0.40 -0.95 0.49]
    // [1.13 -0.08 -0.32]
    // ]


    t1->trunc_();  // In-place
    // [
    // [0.00 -0.00 0.00]
    // [1.00 -0.00 -0.00]
    // ]

    // Other ways 
    Tensor* t2 = t1->trunc(); // returns a new tensor
    Tensor::trunc(t1, t2); // static


Element-wise
-------------

Add
^^^^^
.. doxygenfunction:: Tensor::add(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = Tensor::full({2, 3}, 5.0f);
    // [
    // [5.00 5.00 5.00]
    // [5.00 5.00 5.00]
    // ]

    Tensor* t2 = Tensor::full({2, 3}, 2.0f);
    // [
    // [2.00 2.00 2.00]
    // [2.00 2.00 2.00]
    // ]

    t1->add_(t2);  // In-place
    // [
    // [7.00 7.00 7.00]
    // [7.00 7.00 7.00]
    // ]

    // Other ways
    Tensor* t3 = t1->add(t2);  // returns new tensor
    Tensor::add(t1, t2, t3) //source


Div
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::div(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = Tensor::full({2, 3}, 5.0f);
    // [
    // [5.00 5.00 5.00]
    // [5.00 5.00 5.00]
    // ]

    Tensor* t2 = Tensor::full({2, 3}, 2.0f);
    // [
    // [2.00 2.00 2.00]
    // [2.00 2.00 2.00]
    // ]

    t1->div_(t2);  // In-place
    // [
    // [2.50 2.50 2.50]
    // [2.50 2.50 2.50]
    // ]

    // Other ways
    Tensor* t3 = t1->div(t2);  // returns new tensor
    Tensor::add(t1, t2, t3) //source


Maximum
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::maximum(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2,3});
    // [
    // [-0.25 -1.21 -0.04]
    // [-0.74 0.12 -0.59]
    // ]

    Tensor* t2 = Tensor::randn({2,3});
    // [
    // [0.05 -1.72 -0.50]
    // [0.17 -0.00 0.11]
    // ]

    Tensor* t3 = Tensor::maximum(t1, t2);  // returns a new tensor
    // [
    // [0.05 -1.21 -0.04]
    // [0.17 0.12 0.11]
    // ]

    // Other ways
    Tensor::maximum(t1, t2, t3); // static


Minimum
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::minimum(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = Tensor::randn({2,3});
    // [
    // [-0.25 -0.75 -0.53]
    // [0.29 0.88 1.98]
    // ]

    Tensor* t2 = Tensor::randn({2,3});
    // [
    // [-1.37 0.95 -0.59]
    // [-1.26 1.16 -0.21]
    // ]

    Tensor* t3 = Tensor::minimum(t1, t2);  // returns a new tensor
    // [
    // [-1.37 -0.75 -0.59]
    // [-1.26 0.88 -0.21]
    // ]

    // Other ways
    Tensor::minimum(t1, t2, t3); // static

Mult
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::mult(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = Tensor::full({2, 3}, 5.0f);
    // [
    // [5.00 5.00 5.00]
    // [5.00 5.00 5.00]
    // ]

    Tensor* t2 = Tensor::full({2, 3}, 2.0f);
    // [
    // [2.00 2.00 2.00]
    // [2.00 2.00 2.00]
    // ]

    t1->mult_(t2);  // In-place
    // [
    // [10.00 10.00 10.00]
    // [10.00 10.00 10.00]
    // ]

    // Other ways
    Tensor* t3 = t1->mult(t2);  // returns new tensor
    Tensor::mult(t1, t2, t3) //source

Sub
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sub(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = Tensor::full({2, 3}, 5.0f);
    // [
    // [5.00 5.00 5.00]
    // [5.00 5.00 5.00]
    // ]

    Tensor* t2 = Tensor::full({2, 3}, 2.0f);
    // [
    // [2.00 2.00 2.00]
    // [2.00 2.00 2.00]
    // ]

    t1->sub_(t2);  // In-place
    // [
    // [3.00 3.00 3.00]
    // [3.00 3.00 3.00]
    // ]

    // Other ways
    Tensor* t3 = t1->sub(t2);  // returns new tensor
    Tensor::sub(t1, t2, t3) //source


Reductions
------------------


Argmax
^^^^^^^^
.. doxygenfunction:: Tensor::argmax()


.. code-block:: c++

    Tensor* t1 = Tensor::randn({2, 3});
    // [
    // [2.13 -0.14 -0.15]
    // [1.53 0.96 0.31]
    // ]

    // Global (reduce on all axis)
    int n1 = t1->argmax();
    // 0

    // Reduced on axis 0
    Tensor* t2 = t1->argmax({0}, false); // keepdims==false
    // [0 1 1]

    // Other ways
    int n2 = Tensor::argmax(t1); // static (returns int)
    Tensor* t3 = t1->argmax({0}, true); // axis (returns tensor)


    
Argmin
^^^^^^^^
.. doxygenfunction:: Tensor::argmin()


.. code-block:: c++

    Tensor* t1 = Tensor::randn({2, 3});
    // [
    // [-0.34 -0.13 -0.41]
    // [-1.09 -0.08 -1.85]
    // ]

    // Global (reduce on all axis)
    int n1 = t1->argmin();
    // 5

    // Reduced on axis 0
    Tensor* t2 = t1->argmin({0}, false); // keepdims==false
    t2->print(2);

    // Other ways
    int n2 = Tensor::argmin(t1); // static (returns int)
    Tensor* t3 = t1->argmin({0}, true); // axis (returns tensor)



Max
^^^^^^^^
.. doxygenfunction:: Tensor::max()


.. code-block:: c++

    Tensor* t1 = Tensor::randn({2, 3});
    // [
    // [-0.79 -0.54 0.28]
    // [1.12 0.38 -1.25]
    // ]

    // Global (reduce on all axis)
    float n1 = t1->max();
    // 1.12

    // Reduced on axis 0
    Tensor* t2 = t1->max({0}, false); // keepdims==false
    // [1.12 0.38 0.28]

    // Other ways
    float n2 = Tensor::max(t1); // static (returns float)
    Tensor* t3 = t1->max({0}, true); // axis (returns tensor)


Mean
^^^^^^^^
.. doxygenfunction:: Tensor::mean()


.. code-block:: c++

    Tensor* t1 = Tensor::randn({2, 3});
    // [
    // [-0.01 0.40 -0.07]
    // [1.14 0.13 -0.49]
    // ]

    // Global (reduce on all axis)
    float n1 = t1->mean();
    // 0.18

    // Reduced on axis 0
    Tensor* t2 = t1->mean({0}, false); // keepdims==false
    // [0.57 0.26 -0.28]

    // Other ways
    float n2 = Tensor::mean(t1); // static (returns float)
    Tensor* t3 = t1->mean({0}, true); // axis (returns tensor)


Median
^^^^^^^^
.. doxygenfunction:: Tensor::median()


.. code-block:: c++

    Tensor* t1 = Tensor::randn({2, 3});
    // [
    // [0.24 -0.86 0.36]
    // [0.18 -0.93 -1.07]
    // ]

    // Global (reduce on all axis)
    float n1 = t1->median();
    // -0.34

    // Reduced on axis 0
    Tensor* t2 = t1->median({0}, false); // keepdims==false
    // [0.21 -0.90 -0.35]

    // Reduced on axis 1
    Tensor* t3 = t1->median({1}, false); // keepdims==false
    // [0.24 -0.93]

    // Other ways
    float n2 = Tensor::mean(t1); // static (returns float)
    Tensor* t4 = t1->median({0}, true); // axis (returns tensor)


Min
^^^^^^^^
.. doxygenfunction:: Tensor::min()


.. code-block:: c++

    Tensor* t1 = Tensor::randn({2, 3});
    // [
    // [-1.11 -0.00 0.34]
    // [-0.16 0.13 -0.39]
    // ]

    // Global (reduce on all axis)
    float n1 = t1->min();
    // -1.11

    // Reduced on axis 0
    Tensor* t2 = t1->min({0}, false); // keepdims==false
    // [-1.11 -0.00 -0.39]

    // Other ways
    float n2 = Tensor::min(t1); // static (returns float)
    Tensor* t3 = t1->min({0}, true); // axis (returns tensor)


Mode
^^^^^^^^
.. doxygenfunction:: Tensor::mode()


.. code-block:: c++

    Tensor* t1 = new Tensor({1, 1, 1, 3, 2, 3, 3, 2, 3}, {3, 3});
    // [
    // [1.00 1.00 1.00]
    // [3.00 2.00 3.00]
    // [3.00 2.00 3.00]
    // ]

    // Global (reduce on all axis)
    float n1 = t1->mode();
    // 3

    // Reduced on axis 0
    Tensor* t2 = t1->mode({0}, false); // keepdims==false
    // [3.00 2.00 3.00]

    // Reduced on axis 1
    Tensor* t3 = t1->mode({1}, false); // keepdims==false
    // [1.00 3.00 3.00]

    // Other ways
    float n2 = Tensor::mode(t1); // static (returns float)
    Tensor* t4 = t1->mode({0}, true); // axis (returns tensor)


Norm
^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::norm(vector<int> axis, bool keepdims, string ord = "fro")

Example:

.. code-block:: c++

   Tensor* t1 = new Tensor({1,2,3,4,5,6}, {3, 2});
   // [
   // [1.00 2.00 3.00]
   // [4.00 5.00 6.00]
   // ]

   // Global (reduce on all axis)
   float n1 = t1->norm();
   // 9.53939

   // Reduced on axis 0
   Tensor* t2 = t1->norm({0}, false); // keepdims==false
   // [4.12 5.39 6.71]

   // Other ways
   Tensor::norm(t1, "fro");


Prod
^^^^^^^^
.. doxygenfunction:: Tensor::prod()


.. code-block:: c++

    Tensor* t1 = Tensor::full({2, 3}, 5.0f);
    // [
    // [5.00 5.00 5.00]
    // [5.00 5.00 5.00]
    // ]

    // Global (reduce on all axis)
    float n1 = t1->prod();
    // 15625

    // Reduced on axis 0
    Tensor* t2 = t1->prod({0}, false); // keepdims==false
    // [25.00 25.00 25.00]

    // Reduced on axis 1
    Tensor* t3 = t1->prod({1}, false); // keepdims==false
    // [125.00 125.00]

    // Other ways
    float n2 = Tensor::prod(t1); // static (returns float)
    Tensor* t4 = t1->prod({0}, true); // axis (returns tensor)


Std
^^^^^^^^
.. doxygenfunction:: Tensor::std(bool unbiased = true)


.. code-block:: c++

    Tensor* t1 = Tensor::randn({2, 3});
    // [
    // [0.14 0.29 -0.51]
    // [0.15 -0.82 0.21]
    // ]

    // Global (reduce on all axis)
    float n1 = t1->std();
    // 0.45

    // Reduced on axis 0
    Tensor* t2 = t1->std({0},  false, true); // keepdims==false, unbiased=true
    // [0.01 0.79 0.50]

    // Reduced on axis 1
    Tensor* t3 = t1->std({1}, false, true); // keepdims==false, unbiased=true
    // [0.42 0.58]

    // Other ways
    float n2 = Tensor::std(t1); // static (returns float)

Sum
^^^^^^^^
.. doxygenfunction:: Tensor::sum()


.. code-block:: c++

    Tensor* t1 = Tensor::full({2, 3}, 5.0f);
    // [
    // [5.00 5.00 5.00]
    // [5.00 5.00 5.00]
    // ]

    // Global (reduce on all axis)
    float n1 = t1->sum();
    // 30

    // Reduced on axis 0
    Tensor* t2 = t1->sum({0}, false); // keepdims==false
    // [10.00 10.00 10.00]

    // Reduced on axis 1
    Tensor* t3 = t1->sum({1}, false); // keepdims==false
    // [15.00 15.00]

    // Other ways
    float n2 = Tensor::sum(t1); // static (returns float)
    Tensor* t4 = t1->sum({0}, true); // axis (returns tensor)


Sum Abs
^^^^^^^^
.. doxygenfunction:: Tensor::sum_abs()


.. code-block:: c++

    Tensor* t1 = new Tensor({-5, 5, 5, -5, -5, 5}, {2, 3});
    // [
    // [-5.00 5.00 5.00]
    // [-5.00 -5.00 5.00]
    // ]

    // Global (reduce on all axis)
    float n1 = t1->sum_abs();
    // 30

    // Reduced on axis 0
    Tensor* t2 = t1->sum_abs({0}, false); // keepdims==false
    // [10.00 10.00 10.00]

    // Reduced on axis 1
    Tensor* t3 = t1->sum_abs({1}, false); // keepdims==false
    // [15.00 15.00]

    // Other ways
    float n2 = Tensor::sum_abs(t1); // static (returns float)
    Tensor* t4 = t1->sum_abs({0}, true); // axis (returns tensor)


Var
^^^^^^^^
.. doxygenfunction:: Tensor::var(bool unbiased = true)


.. code-block:: c++

    Tensor* t1 = Tensor::randn({2, 3});
    // [
    // [0.23 -1.09 0.09]
    // [0.09 1.68 -0.06]
    // ]

    // Global (reduce on all axis)
    float n1 = t1->var();
    // 0.78

    // Reduced on axis 0
    Tensor* t2 = t1->var({0},  false, true); // keepdims==false, unbiased=true
    // [0.01 3.84 0.01]

    // Reduced on axis 1
    Tensor* t3 = t1->var({1}, false, true); // keepdims==false, unbiased=true
    // [0.52 0.93]

    // Other ways
    float n2 = Tensor::var(t1); // static (returns int)

