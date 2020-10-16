Mathematical functions
========================

Unary Operations
------------------

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

    t1->add_(2.0); // In-place
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

    Tensor* t1 = Tensor::range(-10.0f, 10.0f);
    // [-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00]

    t1->clamp_(-2.0f, 7.0f); // In-place
    // [-2.00 -2.00 -2.00 -2.00 -1.00 0.00 1.00 2.00 3.00 3.00 3.00]

    // Other ways
    Tensor* t2 = t1->clamp(-2.0f, 7.0f); // returns a new tensor
    Tensor::clamp(t1, t2, -2.0f, 7.0f); // static


clampmax
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::clampmax(float max)

.. code-block:: c++

    Tensor* t1 = new Tensor({0.7753, -0.4702, -0.4599,  1.1899}, {4}, DEV_CPU);
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
.. doxygenfunction:: Tensor::clampmin(float min)
.. doxygenfunction:: Tensor::clampmin(Tensor *A, Tensor *B, float min)

.. code-block:: c++
   
    Tensor* t1 = new Tensor({-0.0299, -2.3184,  2.1593, -0.8883}, {4}, DEV_CPU);
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

.. doxygenfunction:: Tensor::cos()

.. code-block:: c++

    Tensor* t1 = new Tensor({1.4309,  1.2706, -0.8562,  0.9796}, {4}, DEV_CPU);
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

.. doxygenfunction:: Tensor::cosh()

.. code-block:: c++

    Tensor* t1 = new Tensor({0.1632,  1.1835, -0.6979, -0.7325}, {4}, DEV_CPU);
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

.. doxygenfunction:: Tensor::div(float v)

.. code-block:: c++

    blablabla

.. doxygenfunction:: Tensor::div(Tensor *A)


.. code-block:: c++
   
   On the binary section
    


exp
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::exp_()
.. doxygenfunction:: Tensor::exp()
.. doxygenfunction:: Tensor::exp(Tensor *A, Tensor *B)

.. code-block:: c++


    Tensor* t1 = new Tensor({0, 0.69314}, {2}, DEV_CPU);
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
.. doxygenfunction:: Tensor::floor(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = new Tensor({-0.8166,  1.5308, -0.2530, -0.2091}, {4}, DEV_CPU);
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
.. doxygenfunction:: Tensor::inv(float v = 1.0f)
.. doxygenfunction:: Tensor::inv(Tensor *A, Tensor *B, float v = 1.0f)

.. code-block:: c++

    Tensor* t1 = new Tensor({1, 2, 3, 4}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->floor(1);
    // r1 => [1,  0.5, 0.33, 0.25]

    Tensor::floor(t1, r2, 2);
    // r2 => [2,  1, 0.66, 0.5]

    t1->floor_(1);
    // t1 => [1,  0.5, 0.33, 0.25]


log
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::log_()
.. doxygenfunction:: Tensor::log()
.. doxygenfunction:: Tensor::log(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = new Tensor({1, 2, 3, 4}, {4}, DEV_CPU);
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
.. doxygenfunction:: Tensor::log2(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = new Tensor({ 0.8419, 0.8003, 0.9971, 0.5287, 0.0490}, {5}, DEV_CPU);
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
.. doxygenfunction:: Tensor::log10(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = new Tensor({0.5224, 0.9354, 0.7257, 0.1301, 0.2251}, {5}, DEV_CPU);
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
.. doxygenfunction:: Tensor::logn(float n)
.. doxygenfunction:: Tensor::logn(Tensor *A, Tensor *B, float n)

.. code-block:: c++

    Tensor* t1 = new Tensor({0.5224, 0.9354, 0.7257, 0.1301, 0.2251}, {5}, DEV_CPU);
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
.. doxygenfunction:: Tensor::mod(float v)
.. doxygenfunction:: Tensor::mod(Tensor *A, Tensor *B, float v)

.. code-block:: c++


    Tensor* t1 = new Tensor({12, 13, 14, 15, 16}, {5}, DEV_CPU);
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

.. doxygenfunction:: Tensor::mult(float v)
.. doxygenfunction:: Tensor::mult(Tensor *A)


.. code-block:: c++
   
    Tensor* t1 = new Tensor({10, 20, 30, -10}, {4}, DEV_CPU);
    Tensor* t2 = new Tensor({1, 2, 3, 10}, {4}, DEV_CPU);
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
.. doxygenfunction:: Tensor::neg(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = new Tensor({12, 13, 14, 15, 16}, {5}, DEV_CPU);
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
.. doxygenfunction:: Tensor::normalize(float min = 0.0f, float max = 1.0f)
.. doxygenfunction:: Tensor::normalize(Tensor *A, Tensor *B, float min = 0.0f, float max = 1.0f)

.. code-block:: c++

    void normalize_(float min=0.0f, float max=1.0f);
    Tensor* normalize(float min=0.0f, float max=1.0f);
    static void normalize(Tensor*A, Tensor*B, float min=0.0f, float max=1.0f);
    
pow
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::pow_(float)
.. doxygenfunction:: Tensor::pow(float exp)
.. doxygenfunction:: Tensor::pow(Tensor *A, Tensor *B, float exp)

.. code-block:: c++

    void pow_(float exp);
    Tensor* pow(float exp);
    static void pow(Tensor*A, Tensor*B, float min=0.0f, float exp);


powb
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::powb_(float)
.. doxygenfunction:: Tensor::powb(float base)
.. doxygenfunction:: Tensor::powb(Tensor *A, Tensor *B, float base)

.. code-block:: c++

    void powb_(float exp);
    Tensor* powb(float exp);
    static void powb(Tensor*A, Tensor*B, float min=0.0f, float exp);
    
reciprocal
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::reciprocal_()
.. doxygenfunction:: Tensor::reciprocal()
.. doxygenfunction:: Tensor::reciprocal(Tensor *A, Tensor *B)

.. code-block:: c++

    void reciprocal_();
    Tensor* reciprocal();
    static void reciprocal(Tensor*A, Tensor*B);
    
remainder
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::remainder_(float)
.. doxygenfunction:: Tensor::remainder(float v)
.. doxygenfunction:: Tensor::remainder(Tensor *A, Tensor *B, float v)

.. code-block:: c++

    Tensor* t1 = new Tensor({12, 13, 14, 15, 16}, {5}, DEV_CPU);
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
.. doxygenfunction:: Tensor::round(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = new Tensor({2.3, 5.5, 6.1, 7.9, 10.0}, {5}, DEV_CPU);
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
.. doxygenfunction:: Tensor::rsqrt(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = new Tensor({-0.0370,  0.2970,  1.5420, -0.9105}, {4}, DEV_CPU);
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
.. doxygenfunction:: Tensor::sigmoid(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = new Tensor({0.9213,  1.0887, -0.8858, -1.7683}, {4}, DEV_CPU);
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

.. doxygenfunction:: Tensor::sign(float zero_sign = 0.0f)

.. doxygenfunction:: Tensor::sign(Tensor *A, Tensor *B, float zero_sign = 0.0f)

.. code-block:: c++  

    Tensor* t1 = new Tensor({0.7, -1.2, 0., 2.3}, {4}, DEV_CPU);
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
.. doxygenfunction:: Tensor::sin(Tensor *A, Tensor *B)

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
.. doxygenfunction:: Tensor::sinh(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = new Tensor({0.5380, -0.8632, -0.1265,  0.9399}, {4}, DEV_CPU);
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
.. doxygenfunction:: Tensor::sqr(Tensor *A, Tensor *B)

.. code-block:: c++

    void sqr_();
    Tensor* sqr();
    static void sqr(Tensor*A, Tensor*B);
    
sqrt
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sqrt_()
.. doxygenfunction:: Tensor::sqrt()
.. doxygenfunction:: Tensor::sqrt(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = new Tensor({-2.0755,  1.0226,  0.0831,  0.4806}, {4}, DEV_CPU);
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

.. doxygenfunction:: Tensor::sub_(float v)
.. doxygenfunction:: Tensor::sub(float v)
.. doxygenfunction:: Tensor::sub_(Tensor *A)
.. doxygenfunction:: Tensor::sub(Tensor *A)
.. doxygenfunction:: Tensor::sub(Tensor *A, Tensor *B, float v)
.. doxygenfunction:: Tensor::sub(Tensor *A, Tensor *B, Tensor *C)


.. code-block:: c++
   
    Tensor* t1 = new Tensor({10, 20, 30, -10}, {4}, DEV_CPU);
    Tensor* t2 = new Tensor({1, 2, 3, 10}, {4}, DEV_CPU);
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
.. doxygenfunction:: Tensor::tan(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = new Tensor({-1.2027, -1.7687,  0.4412, -1.3856}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->tan();
    // r1 => [-2.5930,  4.9859,  0.4722, -5.3366]

    Tensor::tan(t1, r2);
    // r2 => [-2.5930,  4.9859,  0.4722, -5.3366]

    t1->tan_();
    // t1 => [-2.5930,  4.9859,  0.4722, -5.3366]
    
tanh
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::tanh_()
.. doxygenfunction:: Tensor::tanh()
.. doxygenfunction:: Tensor::tanh(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = new Tensor({0.8986, -0.7279,  1.1745,  0.261}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->tanh();
    // r1 => [0.7156, -0.6218,  0.8257,  0.2553]

    Tensor::tanh(t1, r2);
    // r2 => [0.7156, -0.6218,  0.8257,  0.2553]

    t1->tanh_();
    // t1 => [0.7156, -0.6218,  0.8257,  0.2553]
    
trunc
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::trunc_()
.. doxygenfunction:: Tensor::trunc()
.. doxygenfunction:: Tensor::trunc(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = new Tensor({3.4742,  0.5466, -0.8008, -0.9079}, {4}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;
    
    r1 = t1->trunc();
    // r1 => [3,  0, -0, -0]

    Tensor::trunc(t1, r2);
    // r2 => [3,  0, -0, -0]

    t1->trunc_();
    // t1 => [3,  0, -0, -0]


Binary Operations
-------------------

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

    //Other ways
    Tensor* t3 = t1->add(t2);  // returns new tensor


div
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::div(Tensor *A, Tensor *B)

.. code-block:: c++

    Tensor* t1 = Tensor::eye(3, 3, DEV_CPU);
    // matrix1 => [1 3 3
    //             3 1 3
    //             3 3 1]

    Tensor* t2 = new Tensor(2, 2, 2}, {3}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;

    r1 = Tensor::div(t1, t2); //(new)r1 = t1 / t2
    // r1 => [0.5, 1.5, 1.5
              1.5, 0.5, 1.5
              1.5, 1.5, 0.5] 

    Tensor::add(t1, t2, r2); // C = A / B
    // r2 => [0.5, 1.5, 1.5
              1.5, 0.5, 1.5
              1.5, 1.5, 0.5] 

mult
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::mult(Tensor *A, Tensor *B)
.. doxygenfunction:: Tensor::mult(Tensor *A, Tensor *B, Tensor *C)

.. code-block:: c++

    Tensor* t1 = Tensor::eye(3, 3, DEV_CPU);
    // matrix1 => [1 3 3
    //             3 1 3
    //             3 3 1]

    Tensor* t2 = new Tensor(2, 2, 2}, {3}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;

    r1 = Tensor::mult(t1, t2); //(new)r1 = t1 * t2
    // r1 => [2, 6, 6
              6, 2, 6
              6, 6, 2] 

    Tensor::mult(t1, t2, r2); // C = A * B
    // r2 => [2, 6, 6
              6, 2, 6
              6, 6, 2] 

sub
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sub(Tensor *A, Tensor *B)
.. doxygenfunction:: Tensor::sub(Tensor *A, Tensor *B, Tensor *C)

.. code-block:: c++

    Tensor* t1 = Tensor::eye(3, 3, DEV_CPU);
    // matrix1 => [1 3 3
    //             3 1 3
    //             3 3 1]

    Tensor* t2 = new Tensor(2, 2, 2}, {3}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;

    r1 = Tensor::sub(t1, t2); //(new)r1 = t1 - t2
    // r1 => [-1, 1, 1
              1, -1, 1
              1, 1, -1] 

    Tensor::sub(t1, t2, r2); // C = A - B
    // r2 => [-1, 1, 1
              1, -1, 1
              1, 1, -1]
Reductions
------------------

Apply lower bound
^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::maximum(float v)
.. doxygenfunction:: Tensor::maximum(Tensor *A, float v)
.. doxygenfunction:: Tensor::maximum(Tensor *A, Tensor *B, float v)

.. code-block:: c++
   
    Tensor* maximum(float v);
    static Tensor* maximum(Tensor* A, float v);
    static void maximum(Tensor* A, Tensor* B, float v);


Obtain maximum values
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::maximum(Tensor *A, Tensor *B)
.. doxygenfunction:: Tensor::maximum(Tensor *A, Tensor *B, Tensor *C)

.. code-block:: c++
   
    static Tensor* maximum(Tensor* A, Tensor* B);
    static void maximum(Tensor* A, Tensor* B, Tensor* C);


Apply upper bound
^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::minimum(float v)
.. doxygenfunction:: Tensor::minimum(Tensor *A, float v)
.. doxygenfunction:: Tensor::minimum(Tensor *A, Tensor *B, float v)

.. code-block:: c++
   
    Tensor* minimum(float v);
    static Tensor* minimum(Tensor* A, float v);
    static void minimum(Tensor* A, Tensor* B, float v);


Obtain minumum values
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::minimum(Tensor *A, Tensor *B)
.. doxygenfunction:: Tensor::minimum(Tensor *A, Tensor *B, Tensor *C)

.. code-block:: c++
   
    static Tensor* minimum(Tensor* A, Tensor* B);
    static void minimum(Tensor* A, Tensor* B, Tensor* C);


median
^^^^^^^^
.. doxygenfunction:: Tensor::median()
.. doxygenfunction:: Tensor::median(Tensor *A)


.. code-block:: c++
   

    Tensor* t1 = new Tensor(2, 3, 5, 4, 1}, {5}, DEV_CPU);
    Tensor* r1;
    Tensor* r2;

    float median1 = t1->median();
    // median1 = 3

    float median2 = Tensor::median(t1);
    // median2 = 3

    Tensor* median(vector<int> axis, bool keepdims);
    static void median(Tensor* A, Tensor *B, ReduceDescriptor2 *rd);


max
^^^^^^^^
.. doxygenfunction:: Tensor::max()
.. doxygenfunction:: Tensor::max(Tensor *A)
.. doxygenfunction:: Tensor::max(vector<int> axis, bool keepdims)


.. code-block:: c++

    Tensor* t1 = new Tensor(2, 3, 5, 4, 1}, {5}, DEV_CPU);
    Tensor* r1;

    float max1 = t1->max();
    // max1 = 5

    float max2 = Tensor::max(t1);
    // max2 = 5
    
   
    Tensor* matrix1 = Tensor::eye(3, 3, DEV_CPU);
    // matrix1 => [1 3 3
    //             3 1 3
    //             3 3 1]
    Tensor* max(vector<int> axis, bool keepdims);


argmax
^^^^^^^^
.. doxygenfunction:: Tensor::argmax()
.. doxygenfunction:: Tensor::argmax(Tensor *A)
.. doxygenfunction:: Tensor::argmax(vector<int> axis, bool keepdims)


.. code-block:: c++
   
    Tensor* t1 = new Tensor(2, 3, 5, 4, 1}, {5}, DEV_CPU);
    Tensor* r1;

    float argmax1 = t1->argmax();
    // argmax1 = 2

    float argmax2 = Tensor::argmax(t1);
    // argmax2 = 2
    

    Tensor* argmax(vector<int> axis, bool keepdims);


min
^^^^^^^^
.. doxygenfunction:: Tensor::min()
.. doxygenfunction:: Tensor::min(Tensor *A)
.. doxygenfunction:: Tensor::min(vector<int> axis, bool keepdims)


.. code-block:: c++
   
    Tensor* t1 = new Tensor(2, 3, 5, 4, 1}, {5}, DEV_CPU);
    Tensor* r1;

    float min1 = t1->min();
    // min1 = 1

    float min2 = Tensor::min(t1);
    // min2 = 1

    
    Tensor* min(vector<int> axis, bool keepdims);

    
argmin
^^^^^^^^
.. doxygenfunction:: Tensor::argmin()
.. doxygenfunction:: Tensor::argmin(Tensor *A)
.. doxygenfunction:: Tensor::argmin(vector<int> axis, bool keepdims)


.. code-block:: c++
   
    Tensor* t1 = new Tensor(2, 3, 5, 4, 1}, {5}, DEV_CPU);

    float argmin1 = t1->argmin();
    // argmin1 = 4

    float argmin2 = Tensor::argmin(t1);
    // argmin2 = 4


    Tensor* argmin(vector<int> axis, bool keepdims);


sum
^^^^^^^^
.. doxygenfunction:: Tensor::sum()
.. doxygenfunction:: Tensor::sum(Tensor *A)
.. doxygenfunction:: Tensor::sum(vector<int> axis, bool keepdims)


.. code-block:: c++
   
    Tensor* t1 = new Tensor(2, 3, 5, 4, 1}, {5}, DEV_CPU);

    float sum1 = t1->sum();
    // sum1 = 15

    float sum2 = Tensor::sum(t1);
    // sum2 = 15

    Tensor* sum(vector<int> axis, bool keepdims);


sum_abs
^^^^^^^^
.. doxygenfunction:: Tensor::sum_abs()
.. doxygenfunction:: Tensor::sum_abs(Tensor *A)
.. doxygenfunction:: Tensor::sum_abs(vector<int> axis, bool keepdims)


.. code-block:: c++
   
    Tensor* t1 = new Tensor(-2, -3, -5, -4, -1}, {5}, DEV_CPU);

    float sum1 = t1->sum_abs();
    // sum1 = 15

    float sum2 = Tensor::sum_abs(t1);
    // sum2 = 15



    Tensor* sum_abs(vector<int> axis, bool keepdims);


prod
^^^^^^^^
.. doxygenfunction:: Tensor::prod()
.. doxygenfunction:: Tensor::prod(Tensor *A)
.. doxygenfunction:: Tensor::prod(vector<int> axis, bool keepdims)


.. code-block:: c++
   

    Tensor* t1 = new Tensor(2, 3, 5, 4, 1}, {5}, DEV_CPU);

    float prod1 = t1->prod();
    // prod1 = 120

    float prod2 = Tensor::prod(t1);
    // prod2 = 120

    
    Tensor* prod(vector<int> axis, bool keepdims);


mean
^^^^^^^^
.. doxygenfunction:: Tensor::mean()
.. doxygenfunction:: Tensor::mean(Tensor *A)
.. doxygenfunction:: Tensor::mean(vector<int> axis, bool keepdims)


.. code-block:: c++
   
    Tensor* t1 = new Tensor(2, 3, 5, 4, 1}, {5}, DEV_CPU);

    float mean1 = t1->mean();
    // mean1 = 3

    float mean2 = Tensor::mean(t1);
    // mean2 = 3


    Tensor* mean(vector<int> axis, bool keepdims);


std
^^^^^^^^
.. doxygenfunction:: Tensor::std(bool unbiased = true)
.. doxygenfunction:: Tensor::std(Tensor *A, bool unbiased = true)
.. doxygenfunction:: Tensor::std(vector<int> axis, bool keepdims, bool unbiased = true)


.. code-block:: c++
   
    Tensor* t1 = new Tensor({-0.8166, -1.3802, -0.3560}, {3}, DEV_CPU);

    float std1 = t1->std();
    // std1 = 0.5130

    float std2 = Tensor::std(t1);
    // std2 = 0.5130

    
    Tensor* std(vector<int> axis, bool keepdims, bool unbiased=true);


var
^^^^^^^^
.. doxygenfunction:: Tensor::var(bool unbiased = true)
.. doxygenfunction:: Tensor::var(Tensor *A, bool unbiased = true)
.. doxygenfunction:: Tensor::var(vector<int> axis, bool keepdims, bool unbiased = true)


.. code-block:: c++
   
    Tensor* t1 = new Tensor({-0.3425, -1.2636, -0.4864}, {3}, DEV_CPU);

    float var1 = t1->var();
    // var1 = 0.2455

    float var2 = Tensor::var(t1);
    // var2 = 0.2455


    Tensor* var(vector<int> axis, bool keepdims, bool unbiased=true);


mode
^^^^^^^^
.. doxygenfunction:: Tensor::mode()
.. doxygenfunction:: Tensor::mode(Tensor *A)
.. doxygenfunction:: Tensor::mode(vector<int> axis, bool keepdims)


.. code-block:: c++
   
    Tensor* t1 = new Tensor({2, 2, 1, 5 ,3}, {5}, DEV_CPU);

    float mode1 = t1->mode();
    // mode1 = 2

    float mode2 = Tensor::mode(t1);
    // mode2 = 2


    Tensor* mode(vector<int> axis, bool keepdims);


Matrix Operations
--------------------

sum
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sum2D_rowwise

.. doxygenfunction:: Tensor::sum2D_colwise

.. code-block:: c++

    Tensor* matrix1 = Tensor::eye(3, 3, DEV_CPU);
    // matrix1 => [1 3 3
    //             3 1 3
    //             3 3 1]

    Tensor* matrix2 = Tensor::identity(3, DEV_CPU);
    // matrix2 => [1 0 0
    //             0 1 0
    //             0 0 1]

    Tensor* r1;
    Tensor* r2;

    Tensor::sum2D_rowwise(matrix1, matrix2, r1);
    // r1 => [2 3 3
    //        3 2 3
    //        3 3 2]

    Tensor::sum2D_colwise(matrix1, matrix2, r2);
    // r2 => [2 3 3
    //        3 2 3
    //        3 3 2]



mult
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::mult2D

.. code-block:: c++

    Tensor* matrix1 = Tensor::eye(3, 3, DEV_CPU);
    // matrix1 => [1 3 3
    //             3 1 3
    //             3 3 1]

    Tensor* matrix2 = Tensor::identity(3, DEV_CPU);
    // matrix2 => [1 0 0
    //             0 1 0
    //             0 0 1]

    Tensor* matrix3 = Tensor::identity(3, DEV_CPU);
    // matrix2 => [1 0 0
    //             0 1 0
    //             0 0 1]


    Tensor::mult2D(matrix1, 0 matrix2, 1, matrix3, 1);
    // matrix3 => [2 0 0
    //             0 2 0
    //             0 0 2]

   
