Mathematical functions
========================

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/docs/markdown/eddl_progress_tensor.md


Unary Operations
------------------

abs
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::abs_
.. doxygenfunction:: Tensor::abs

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({-1, -2, 3}, {3}, DEV_CPU);
    
    Tensor* r1 = t1->abs();
    // r1 => [1, 2, 3]

    t1->abs_();
    // t1 => [1, 2, 3]
    
acos
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::acos_
.. doxygenfunction:: Tensor::acos()
.. doxygenfunction:: Tensor::acos(Tensor*, Tensor*)


.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({0.3348, -0.5889,  0.2005, -0.1584}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->acos();
    // r1 => [1.2294,  2.2004,  1.3690,  1.7298]

    Tensor::acos(t1, r2);
    // r2 => [1.2294,  2.2004,  1.3690,  1.7298]

    t1->acos_();
    // t1 => [1.2294,  2.2004,  1.3690,  1.7298]

    
add
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::add_(float)
.. doxygenfunction:: Tensor::add(float)
.. doxygenfunction:: Tensor::add_(Tensor*)
.. doxygenfunction:: Tensor::add(Tensor*)
.. doxygenfunction:: Tensor::add(Tensor*, Tensor*, float)
.. doxygenfunction:: Tensor::add(float, Tensor*, float, Tensor*, Tensor*, int)


.. code-block:: c++
   
    Tensor* t1 = new Tensor::Tensor({10, 20, 30, -10}, {4}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({1, 2, 3, 4}, {4}, DEV_CPU);
    Tensor* r3;
    Tensor* r4;
    
    Tensor* r1 = t1->add(20.0);
    // r1 => [30, 40, 50, 10]
    
    Tensor* r2 = t1->add(t2);  // this = this .+ A
    // r2 => [11, 22, 33, -6]

    Tensor::add(t1, r3, 20.0); // B = A + v
    // r3 => [30, 40, 50, 10]

    Tensor::add(1, t1, 2, t2, r4, 1); // C = a*A+b*B
    // r4 => [12, 24, 36, -2]

    
    t2->add_(t1);  // this = this .+ A
    // t2 => [11, 22, 33, -6]

    t1->add_(20);
    // t1 => [30, 40, 50, 10]

    


asin
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::asin_
.. doxygenfunction:: Tensor::asin()
.. doxygenfunction:: Tensor::asin(Tensor*, Tensor*)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({-0.5962,  1.4985, -0.4396,  1.4525}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->asin();
    // r1 => [-0.6387,     nan, -0.4552,     nan]

    Tensor::asin(t1, r2);
    // r2 => [-0.6387,     nan, -0.4552,     nan]

    t1->asin_();
    // t1 => [-0.6387,     nan, -0.4552,     nan]

    
atan
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::atan_()
.. doxygenfunction:: Tensor::atan()
.. doxygenfunction:: Tensor::atan(Tensor*, Tensor*)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({0.2341, 0.2539, -0.6256, -0.6448}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->atan();
    // r1 => [0.2299, 0.2487, -0.5591, -0.5727]

    Tensor::atan(t1, r2);
    // r2 => [0.2299, 0.2487, -0.5591, -0.5727]

    t1->atan_();
    // t1 => [0.2299, 0.2487, -0.5591, -0.5727]


    
ceil
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::ceil_()
.. doxygenfunction:: Tensor::ceil()
.. doxygenfunction:: Tensor::ceil(Tensor*, Tensor*)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({-0.6341, -1.4208, -1.0900,  0.5826}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->ceil();
    // r1 => [0, -1, -1, 1]

    Tensor::ceil(t1, r2);
    // r2 => [0, -1, -1, 1]

    t1->ceil_();
    // t1 => [0, -1, -1, 1]


clamp
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::clamp_(float, float)
.. doxygenfunction:: Tensor::clamp(float, float)
.. doxygenfunction:: Tensor::clamp(Tensor*, Tensor*, float, float)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({-1.7120,  0.1734, -0.0478, -0.0922}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->clamp(-0.5, 0.5);
    // r1 => [-0.5000,  0.1734, -0.0478, -0.0922]

    Tensor::clamp(t1, r2, -0.5, 0.5);
    // r2 => [-0.5000,  0.1734, -0.0478, -0.0922]

    t1->clamp_(-0.5, 0.5);
    // t1 => [-0.5000,  0.1734, -0.0478, -0.0922]




    
clampmax
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::clampmax_(float)
.. doxygenfunction:: Tensor::clampmax(float)
.. doxygenfunction:: Tensor::clampmax(Tensor*, Tensor*, float)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({0.7753, -0.4702, -0.4599,  1.1899}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->clampmax(0.5);
    // r1 => [0.5000, -0.4702, -0.4599,  0.5000]

    Tensor::clampmax(t1, r2, 0.5);
    // r2 => [0.5000, -0.4702, -0.4599,  0.5000]

    t1->clampmax_(0.5);
    // t1 => [0.5000, -0.4702, -0.4599,  0.5000]
   

    
clampmin
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::clampmin_(float)
.. doxygenfunction:: Tensor::clampmin(float)
.. doxygenfunction:: Tensor::clampmin(Tensor*, Tensor*, float)

.. code-block:: c++
   
    Tensor* t1 = new Tensor::Tensor({-0.0299, -2.3184,  2.1593, -0.8883}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->clampmin(0.5);
    // r1 => [0.5000,  0.5000,  2.1593,  0.5000]

    Tensor::clampmin(t1, r2, 0.5);
    // r2 => [0.5000,  0.5000,  2.1593,  0.5000]

    t1->clampmin_(0.5);
    // t1 => [0.5000,  0.5000,  2.1593,  0.5000]

    
cos
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cos_()
.. doxygenfunction:: Tensor::cos()
.. doxygenfunction:: Tensor::cos(Tensor*, Tensor*)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({1.4309,  1.2706, -0.8562,  0.9796}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->cos();
    // r1 => [0.1395,  0.2957,  0.6553,  0.5574]

    Tensor::cos(t1, r2);
    // r2 => [0.1395,  0.2957,  0.6553,  0.5574]

    t1->cos_();
    // t1 => [0.1395,  0.2957,  0.6553,  0.5574]

    
cosh
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cosh_()
.. doxygenfunction:: Tensor::cosh()
.. doxygenfunction:: Tensor::cosh(Tensor*, Tensor*)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({0.1632,  1.1835, -0.6979, -0.7325}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->cosh();
    // r1 => [1.0133,  1.7860,  1.2536,  1.2805]

    Tensor::cosh(t1, r2);
    // r2 => [1.0133,  1.7860,  1.2536,  1.2805]

    t1->cosh_();
    // t1 => [1.0133,  1.7860,  1.2536,  1.2805]
  
    
div
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::div_(float)
.. doxygenfunction:: Tensor::div(float)
.. doxygenfunction:: Tensor::div_(Tensor*)
.. doxygenfunction:: Tensor::div(Tensor*)
.. doxygenfunction:: Tensor::div(Tensor*, Tensor*, float)


.. code-block:: c++
   
    Tensor* t1 = new Tensor::Tensor({10, 20, 30, -10}, {4}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({1, 2, 3, 10}, {4}, DEV_CPU);
    Tensor* r3;
    Tensor* r4;
    
    Tensor* r1 = t1->div(10.0);
    // r1 => [1, 2, 3, -1]
    
    Tensor* r2 = t1->div(t2);  // this = this ./ A
    // r2 => [10, 10, 10, -1]

    Tensor::div(t1, r3, 10.0); // B = A / v
    // r3 => [1, 2, 3, -1]
    
    t2->div_(t1);  // this = this ./ A
    // t2 => [0.1, 0.1, 0.1, -1]

    t1->div_(20);
    // t1 => [1, 2, 3, -1]
    

el_div
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::el_div

.. code-block:: c++


    Tensor* t1 = new Tensor::Tensor({10, 20, 30, -10, 10, 20, 30, -10}, {2,4}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({1, 2, 3, 10}, {4}, DEV_CPU);
    Tensor* r3;

    Tensor::el_div(t1, t2, r3, 1);
    // r3 => [10, 10, 10, -1
    //        10, 10, 10, -1]


el_mult
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::el_mult

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({10, 20, 30, -10, 10, 20, 30, -10}, {2,4}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({1, 2, 3, 10}, {4}, DEV_CPU);
    Tensor* r3;

    Tensor::el_mult(t1, t2, r3, 1);
    // r3 => [10, 40, 90, -100
    //        10, 40, 90, -100]

exp
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::exp_()
.. doxygenfunction:: Tensor::exp()
.. doxygenfunction:: Tensor::exp(Tensor*, Tensor*)

.. code-block:: c++


    Tensor* t1 = new Tensor::Tensor({0, 0.69314}, {2}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->exp();
    // r1 => [1, 2]

    Tensor::exp(t1, r2);
    // r2 => [1, 2]

    t1->exp_();
    // t1 => [1, 2]


floor
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::floor_()
.. doxygenfunction:: Tensor::floor()
.. doxygenfunction:: Tensor::floor(Tensor*, Tensor*)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({-0.8166,  1.5308, -0.2530, -0.2091}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->floor();
    // r1 => [-1,  1, -1, -1]

    Tensor::floor(t1, r2);
    // r2 => [-1,  1, -1, -1]

    t1->floor_();
    // t1 => [-1,  1, -1, -1]


inv
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::inv_(float)
.. doxygenfunction:: Tensor::inv(float)
.. doxygenfunction:: Tensor::inv(Tensor*, Tensor*, float)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({1, 2, 3, 4}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->floor(1);
    // r1 => [1,  0.5, 0.33, 0.25]

    Tensor::floor(t1, r2, 2);
    // r2 => [2,  1, 0.66, 0.5]

    t1->floor_(1);
    // t1 => [1,  0.5, 0.33, 0.25]


inc
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::inc

.. code-block:: c++

    static void inc(Tensor*A, Tensor*B);
    
log
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::log_()
.. doxygenfunction:: Tensor::log()
.. doxygenfunction:: Tensor::log(Tensor*, Tensor*)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({1, 2, 3, 4}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->log();
    // r1 => [0,  0.693147, 1.098612, 1.386294]

    Tensor::log(t1, r2);
    // r2 => [0,  0.693147, 1.098612, 1.386294]

    t1->log_();
    // t1 => [0,  0.693147, 1.098612, 1.386294]

    
log2
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::log2_()
.. doxygenfunction:: Tensor::log2()
.. doxygenfunction:: Tensor::log2(Tensor*, Tensor*)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({ 0.8419, 0.8003, 0.9971, 0.5287, 0.0490}, {5}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->log2();
    // r1 => [-0.2483, -0.3213, -0.0042, -0.9196, -4.3504]

    Tensor::log2(t1, r2);
    // r2 => [-0.2483, -0.3213, -0.0042, -0.9196, -4.3504]

    t1->log2_();
    // t1 => [-0.2483, -0.3213, -0.0042, -0.9196, -4.3504]
  
    
log10
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::log10_()
.. doxygenfunction:: Tensor::log10()
.. doxygenfunction:: Tensor::log10(Tensor*, Tensor*)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({0.5224, 0.9354, 0.7257, 0.1301, 0.2251}, {5}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->log10();
    // r1 => [-0.2820, -0.0290, -0.1392, -0.8857, -0.6476]

    Tensor::log10(t1, r2);
    // r2 => [-0.2820, -0.0290, -0.1392, -0.8857, -0.6476]

    t1->log10_();
    // t1 => [-0.2820, -0.0290, -0.1392, -0.8857, -0.6476]
    
    
logn
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logn_(float)
.. doxygenfunction:: Tensor::logn(float)
.. doxygenfunction:: Tensor::logn(Tensor*, Tensor*, float)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({0.5224, 0.9354, 0.7257, 0.1301, 0.2251}, {5}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->logn(10.0);
    // r1 => [-0.2820, -0.0290, -0.1392, -0.8857, -0.6476]

    Tensor::log10(t1, r2, 10);
    // r2 => [-0.2820, -0.0290, -0.1392, -0.8857, -0.6476]

    t1->logn_(10);
    // t1 => [-0.2820, -0.0290, -0.1392, -0.8857, -0.6476]

    
mod
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::mod_(float)
.. doxygenfunction:: Tensor::mod(float)
.. doxygenfunction:: Tensor::mod(Tensor*, Tensor*, float)

.. code-block:: c++


    Tensor* t1 = new Tensor::Tensor({12, 13, 14, 15, 16}, {5}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->mod(10.0);
    // r1 => [2, 3, 4, 5, 6]

    Tensor::log10(t1, r2, 10);
    // r2 => [2, 3, 4, 5, 6]

    t1->logn_(10);
    // t1 => [2, 3, 4, 5, 6]
    
mult
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::mult_(float)
.. doxygenfunction:: Tensor::mult(float)
.. doxygenfunction:: Tensor::mult_(Tensor*)
.. doxygenfunction:: Tensor::mult(Tensor*)
.. doxygenfunction:: Tensor::mult(Tensor*, Tensor*, float)


.. code-block:: c++
   
    Tensor* t1 = new Tensor::Tensor({10, 20, 30, -10}, {4}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({1, 2, 3, 10}, {4}, DEV_CPU);
    Tensor* r3;
    Tensor* r4;
    
    Tensor* r1 = t1->mult(10.0);
    // r1 => [100, 200, 300, -100]
    
    Tensor* r2 = t1->mult(t2);  // this = this .* A
    // r2 => [10, 40, 90, -100]

    Tensor::mult(t1, r3, 10.0); // B = A * v
    // r3 => [100, 200, 300, -100]
    
    t2->mult_(t1);  // this = this .* A
    // t2 => [10, 40, 90, -100]

    t1->mult_(10);
    // t1 => [1, 2, 3, -1]
    
neg
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::neg_()
.. doxygenfunction:: Tensor::neg()
.. doxygenfunction:: Tensor::neg(Tensor*, Tensor*)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({12, 13, 14, 15, 16}, {5}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->neg();
    // r1 => [-12, -13, -14, -15, -16]

    Tensor::neg(t1, r2);
    // r2 => [-12, -13, -14, -15, -16]

    t1->neg_();
    // t1 => [-12, -13, -14, -15, -16]

normalize
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::normalize_(float, float)
.. doxygenfunction:: Tensor::normalize(float, float)
.. doxygenfunction:: Tensor::normalize(Tensor*, Tensor*, float, float)

.. code-block:: c++

    void normalize_(float min=0.0f, float max=1.0f);
    Tensor* normalize(float min=0.0f, float max=1.0f);
    static void normalize(Tensor*A, Tensor*B, float min=0.0f, float max=1.0f);
    
pow
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::pow_(float)
.. doxygenfunction:: Tensor::pow(float)
.. doxygenfunction:: Tensor::pow(Tensor*, Tensor*, float)

.. code-block:: c++

    void pow_(float exp);
    Tensor* pow(float exp);
    static void pow(Tensor*A, Tensor*B, float min=0.0f, float exp);


powb
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::powb_(float)
.. doxygenfunction:: Tensor::powb(float)
.. doxygenfunction:: Tensor::powb(Tensor*, Tensor*, float)

.. code-block:: c++

    void powb_(float exp);
    Tensor* powb(float exp);
    static void powb(Tensor*A, Tensor*B, float min=0.0f, float exp);
    
reciprocal
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::reciprocal_()
.. doxygenfunction:: Tensor::reciprocal()
.. doxygenfunction:: Tensor::reciprocal(Tensor*, Tensor*)

.. code-block:: c++

    void reciprocal_();
    Tensor* reciprocal();
    static void reciprocal(Tensor*A, Tensor*B);
    
remainder
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::remainder_(float)
.. doxygenfunction:: Tensor::remainder(float)
.. doxygenfunction:: Tensor::remainder(Tensor*, Tensor*, float)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({12, 13, 14, 15, 16}, {5}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->remainder(10.0);
    // r1 => [2, 3, 4, 5, 6]

    Tensor::remainder(t1, r2, 10);
    // r2 => [2, 3, 4, 5, 6]

    t1->remainder_(10);
    // t1 => [2, 3, 4, 5, 6]
    
    
round
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::round_()
.. doxygenfunction:: Tensor::round()
.. doxygenfunction:: Tensor::round(Tensor*, Tensor*)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({2.3, 5.5, 6.1, 7.9, 10.0}, {5}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->round();
    // r1 => [2, 6, 6, 8, 10]

    Tensor::round(t1, r2);
    // r2 => [2, 6, 6, 8, 10]

    t1->round_();
    // t1 => [2, 6, 6, 8, 10]
    
rsqrt
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::rsqrt_()
.. doxygenfunction:: Tensor::rsqrt()
.. doxygenfunction:: Tensor::rsqrt(Tensor*, Tensor*)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({-0.0370,  0.2970,  1.5420, -0.9105}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->rsqrt();
    // r1 => [nan,  1.8351,  0.8053,   nan]

    Tensor::rsqrt(t1, r2);
    // r2 => [nan,  1.8351,  0.8053,   nan]

    t1->rsqrt_();
    // t1 => [nan,  1.8351,  0.8053,   nan]
sigmoid
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sigmoid_()
.. doxygenfunction:: Tensor::sigmoid()
.. doxygenfunction:: Tensor::sigmoid(Tensor*, Tensor*)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({0.9213,  1.0887, -0.8858, -1.7683}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->sigmoid();
    // r1 => [0.7153,  0.7481,  0.2920,  0.1458]

    Tensor::sigmoid(t1, r2);
    // r2 => [0.7153,  0.7481,  0.2920,  0.1458]

    t1->sigmoid_();
    // t1 => [0.7153,  0.7481,  0.2920,  0.1458]
    
sign
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sign(float)

.. doxygenfunction:: Tensor::sign(Tensor*, Tensor*, float)

.. code-block:: c++  

    Tensor* t1 = new Tensor::Tensor({0.7, -1.2, 0., 2.3}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->sign(5);
    // r1 => [1, -1,  5,  1]

    Tensor::sign(t1, r2, 5);
    // r2 => [1, -1,  5,  1]

    t1->sign_();
    // t1 => [1, -1,  5,  1]
    

sin
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sin_()
.. doxygenfunction:: Tensor::sin()
.. doxygenfunction:: Tensor::sin(Tensor*, Tensor*)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({-0.5461,  0.1347, -2.7266, -0.2746}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->sin();
    // r1 => [-0.5194,  0.1343, -0.4032, -0.2711]

    Tensor::sin(t1, r2);
    // r2 => [-0.5194,  0.1343, -0.4032, -0.2711]

    t1->sin_();
    // t1 => [-0.5194,  0.1343, -0.4032, -0.2711]
    
sinh
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sinh_()
.. doxygenfunction:: Tensor::sinh()
.. doxygenfunction:: Tensor::sinh(Tensor*, Tensor*)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({0.5380, -0.8632, -0.1265,  0.9399}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->sinh();
    // r1 => [0.5644, -0.9744, -0.1268,  1.0845]

    Tensor::sinh(t1, r2);
    // r2 => [0.5644, -0.9744, -0.1268,  1.0845]

    t1->sinh_();
    // t1 => [0.5644, -0.9744, -0.1268,  1.0845]
    
sqr
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sqr_()
.. doxygenfunction:: Tensor::sqr()
.. doxygenfunction:: Tensor::sqr(Tensor*, Tensor*)

.. code-block:: c++

    void sqr_();
    Tensor* sqr();
    static void sqr(Tensor*A, Tensor*B);
    
sqrt
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sqrt_()
.. doxygenfunction:: Tensor::sqrt()
.. doxygenfunction:: Tensor::sqrt(Tensor*, Tensor*)

.. code-block:: c++

    Tensor* t1 = new Tensor::Tensor({-2.0755,  1.0226,  0.0831,  0.4806}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->sqrt();
    // r1 => [nan,  1.0112,  0.2883,  0.6933]

    Tensor::sqrt(t1, r2);
    // r2 => [nan,  1.0112,  0.2883,  0.6933]

    t1->sqrt_();
    // t1 => [nan,  1.0112,  0.2883,  0.6933]
    
sub
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sub_(float)
.. doxygenfunction:: Tensor::sub(float)
.. doxygenfunction:: Tensor::sub_(Tensor*)
.. doxygenfunction:: Tensor::sub(Tensor*)
.. doxygenfunction:: Tensor::sub(Tensor*, Tensor*, float)


.. code-block:: c++
   
    Tensor* t1 = new Tensor::Tensor({10, 20, 30, -10}, {4}, DEV_CPU);
    Tensor* t2 = new Tensor::Tensor({1, 2, 3, 10}, {4}, DEV_CPU);
    Tensor* r3;
    Tensor* r4;
    
    Tensor* r1 = t1->sub(10.0);
    // r1 => [0, 10, 20, -20]
    
    Tensor* r2 = t1->sub(t2);  // this = this .- A
    // r2 => [9, 18, 27, -20]

    Tensor::sub(t1, r3, 10.0); // B = A - v
    // r3 => [0, 10, 20, -20]
    
    t2->sub_(t1);  // this = this .- A
    // t2 => [-9, -18, -27, 0]

    t1->sub_(10);
    // t1 => 0, 10, 20, -20]
    

    
tan
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::tan_()
.. doxygenfunction:: Tensor::tan()
.. doxygenfunction:: Tensor::tan(Tensor*, Tensor*)

.. code-block:: c++

    void tan_();
    Tensor* tan();
    static void tan(Tensor*A, Tensor*B);
    
tanh
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::tanh_()
.. doxygenfunction:: Tensor::tanh()
.. doxygenfunction:: Tensor::tanh(Tensor*, Tensor*)

.. code-block:: c++

    void tanh_();
    Tensor* tanh();
    static void tanh(Tensor*A, Tensor*B);
    
trunc
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::trunc_()
.. doxygenfunction:: Tensor::trunc()
.. doxygenfunction:: Tensor::trunc(Tensor*, Tensor*)

.. code-block:: c++

    void trunc_();
    Tensor* trunc();
    static void trunc(Tensor*A, Tensor*B);


Binary Operations
-------------------

add
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::add(Tensor*, Tensor*)
.. doxygenfunction:: Tensor::add(Tensor*, Tensor*, Tensor*)

.. code-block:: c++

    static Tensor* add(Tensor*A, Tensor*B); // (new)C = A + B
    static void add(Tensor*A, Tensor*B, Tensor*C); // C = A + B


div
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::div(Tensor*, Tensor*)
.. doxygenfunction:: Tensor::div(Tensor*, Tensor*, Tensor*)

.. code-block:: c++

    static Tensor* div(Tensor*A, Tensor*B); // (new)C = A / B
    static void div(Tensor*A, Tensor*B, Tensor*C); // C = A / B

mult
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::mult(Tensor*, Tensor*)
.. doxygenfunction:: Tensor::mult(Tensor*, Tensor*, Tensor*)

.. code-block:: c++

    static Tensor* mult(Tensor*A, Tensor*B); // (new)C = A * B
    static void mult(Tensor*A, Tensor*B, Tensor*C); // C = A * B

sub
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sub(Tensor*, Tensor*)
.. doxygenfunction:: Tensor::sub(Tensor*, Tensor*, Tensor*)

.. code-block:: c++

    static Tensor* sub(Tensor*A, Tensor*B); // (new)C = A - B
    static void sub(Tensor*A, Tensor*B, Tensor*C); // C = A - B

Reductions
------------------

Apply lower bound
^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::maximum(float)
.. doxygenfunction:: Tensor::maximum(Tensor*, float)
.. doxygenfunction:: Tensor::maximum(Tensor*, Tensor*, float)

.. code-block:: c++
   
    Tensor* maximum(float v);
    static Tensor* maximum(Tensor* A, float v);
    static void maximum(Tensor* A, Tensor* B, float v);


Obtain maximum values
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::maximum(Tensor*, Tensor*)
.. doxygenfunction:: Tensor::maximum(Tensor*, Tensor*, Tensor*)

.. code-block:: c++
   
    static Tensor* maximum(Tensor* A, Tensor* B);
    static void maximum(Tensor* A, Tensor* B, Tensor* C);


Apply upper bound
^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::minimum(float)
.. doxygenfunction:: Tensor::minimum(Tensor*, float)
.. doxygenfunction:: Tensor::minimum(Tensor*, Tensor*, float)

.. code-block:: c++
   
    Tensor* minimum(float v);
    static Tensor* minimum(Tensor* A, float v);
    static void minimum(Tensor* A, Tensor* B, float v);


Obtain minumum values
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::minimum(Tensor*, Tensor*)
.. doxygenfunction:: Tensor::minimum(Tensor*, Tensor*, Tensor*)

.. code-block:: c++
   
    static Tensor* minimum(Tensor* A, Tensor* B);
    static void minimum(Tensor* A, Tensor* B, Tensor* C);


median
^^^^^^^^
.. doxygenfunction:: Tensor::median()
.. doxygenfunction:: Tensor::median(Tensor*)


.. code-block:: c++
   
    float median();
    static float median(Tensor* A);


max
^^^^^^^^
.. doxygenfunction:: Tensor::max()
.. doxygenfunction:: Tensor::max(Tensor*)
.. doxygenfunction:: Tensor::max(vector<int>, bool)


.. code-block:: c++
   
    float max();
    static float max(Tensor* A);
    Tensor* max(vector<int> axis, bool keepdims);


argmax
^^^^^^^^
.. doxygenfunction:: Tensor::argmax()
.. doxygenfunction:: Tensor::argmax(Tensor*)
.. doxygenfunction:: Tensor::argmax(vector<int>, bool)


.. code-block:: c++
   
    float argmax();
    static float argmax(Tensor* A);
    Tensor* argmax(vector<int> axis, bool keepdims);


min
^^^^^^^^
.. doxygenfunction:: Tensor::min()
.. doxygenfunction:: Tensor::min(Tensor*)
.. doxygenfunction:: Tensor::min(vector<int>, bool)


.. code-block:: c++
   
    float min();
    static float min(Tensor* A);
    Tensor* min(vector<int> axis, bool keepdims);

    
argmin
^^^^^^^^
.. doxygenfunction:: Tensor::argmin()
.. doxygenfunction:: Tensor::argmin(Tensor*)
.. doxygenfunction:: Tensor::argmin(vector<int>, bool)


.. code-block:: c++
   
    float argmin();
    static float argmin(Tensor* A);
    Tensor* argmin(vector<int> axis, bool keepdims);


sum
^^^^^^^^
.. doxygenfunction:: Tensor::sum()
.. doxygenfunction:: Tensor::sum(Tensor*)
.. doxygenfunction:: Tensor::sum(vector<int>, bool)


.. code-block:: c++
   
    float sum();
    static float sum(Tensor* A);
    Tensor* sum(vector<int> axis, bool keepdims);


sum_abs
^^^^^^^^
.. doxygenfunction:: Tensor::sum_abs()
.. doxygenfunction:: Tensor::sum_abs(Tensor*)
.. doxygenfunction:: Tensor::sum_abs(vector<int>, bool)


.. code-block:: c++
   
    float sum_abs();
    static float sum_abs(Tensor* A);
    Tensor* sum_abs(vector<int> axis, bool keepdims);


prod
^^^^^^^^
.. doxygenfunction:: Tensor::prod()
.. doxygenfunction:: Tensor::prod(Tensor*)
.. doxygenfunction:: Tensor::prod(vector<int>, bool)


.. code-block:: c++
   
    float prod();
    static float prod(Tensor* A);
    Tensor* prod(vector<int> axis, bool keepdims);


mean
^^^^^^^^
.. doxygenfunction:: Tensor::mean()
.. doxygenfunction:: Tensor::mean(Tensor*)
.. doxygenfunction:: Tensor::mean(vector<int>, bool)


.. code-block:: c++
   
    float mean();
    static float mean(Tensor* A);
    Tensor* mean(vector<int> axis, bool keepdims);


std
^^^^^^^^
.. doxygenfunction:: Tensor::std(bool)
.. doxygenfunction:: Tensor::std(Tensor*, bool)
.. doxygenfunction:: Tensor::std(vector<int>, bool, bool)


.. code-block:: c++
   
    float std(bool unbiased=true);
    static float std(Tensor* A, bool unbiased=true);
    Tensor* std(vector<int> axis, bool keepdims, bool unbiased=true);


var
^^^^^^^^
.. doxygenfunction:: Tensor::var(bool)
.. doxygenfunction:: Tensor::var(Tensor*, bool)
.. doxygenfunction:: Tensor::var(vector<int>, bool, bool)


.. code-block:: c++
   
    float var(bool unbiased=true);
    static float var(Tensor* A, bool unbiased=true);
    Tensor* var(vector<int> axis, bool keepdims, bool unbiased=true);


mode
^^^^^^^^
.. doxygenfunction:: Tensor::mode()
.. doxygenfunction:: Tensor::mode(Tensor*)
.. doxygenfunction:: Tensor::mode(vector<int>, bool)


.. code-block:: c++
   
    float mode();
    static float mode(Tensor* A);
    Tensor* mode(vector<int> axis, bool keepdims);


Matrix Operations
--------------------

sum
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sum2D_rowwise

.. doxygenfunction:: Tensor::sum2D_colwise

.. code-block:: c++
   
    static void sum2D_rowwise(Tensor*A, Tensor*B, Tensor*C);
    static void sum2D_colwise(Tensor*A, Tensor*B, Tensor*C);


mult
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::mult2D

.. code-block:: c++
   
    static void mult2D(Tensor*A, int tA, Tensor*B, int tB, Tensor*C, int incC);
