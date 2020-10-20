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

    blablabla
   

    
clampmin
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::clampmin_(float)
.. doxygenfunction:: Tensor::clampmin(float min)
.. doxygenfunction:: Tensor::clampmin(Tensor *A, Tensor *B, float min)

.. code-block:: c++

    blablabla

    
cos
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cos()

.. code-block:: c++

    blablabla

    
cosh
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cosh()

.. code-block:: c++

    blablabla
  
    
div
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::div(float v)

.. code-block:: c++

    blablabla
    


exp
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::exp_()
.. doxygenfunction:: Tensor::exp()
.. doxygenfunction:: Tensor::exp(Tensor *A, Tensor *B)

.. code-block:: c++

    blablabla


floor
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::floor_()
.. doxygenfunction:: Tensor::floor()
.. doxygenfunction:: Tensor::floor(Tensor *A, Tensor *B)

.. code-block:: c++

    blablabla


inv
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::inv_(float)
.. doxygenfunction:: Tensor::inv(float v = 1.0f)
.. doxygenfunction:: Tensor::inv(Tensor *A, Tensor *B, float v = 1.0f)

.. code-block:: c++

    blablabla


log
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::log_()
.. doxygenfunction:: Tensor::log()
.. doxygenfunction:: Tensor::log(Tensor *A, Tensor *B)

.. code-block:: c++

    blablabla

    
log2
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::log2_()
.. doxygenfunction:: Tensor::log2()
.. doxygenfunction:: Tensor::log2(Tensor *A, Tensor *B)

.. code-block:: c++

    blablabla
  
    
log10
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::log10_()
.. doxygenfunction:: Tensor::log10()
.. doxygenfunction:: Tensor::log10(Tensor *A, Tensor *B)

.. code-block:: c++

    blablabla
    
    
logn
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logn_(float)
.. doxygenfunction:: Tensor::logn(float n)
.. doxygenfunction:: Tensor::logn(Tensor *A, Tensor *B, float n)

.. code-block:: c++

    blablabla

    
mod
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::mod_(float)
.. doxygenfunction:: Tensor::mod(float v)
.. doxygenfunction:: Tensor::mod(Tensor *A, Tensor *B, float v)

.. code-block:: c++

    blablabla

    
mult
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::mult(float v)
.. doxygenfunction:: Tensor::mult(Tensor *A)


.. code-block:: c++

    blablabla
    
neg
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::neg_()
.. doxygenfunction:: Tensor::neg()
.. doxygenfunction:: Tensor::neg(Tensor *A, Tensor *B)

.. code-block:: c++

    blablabla

normalize
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::normalize_(float, float)
.. doxygenfunction:: Tensor::normalize(float min = 0.0f, float max = 1.0f)
.. doxygenfunction:: Tensor::normalize(Tensor *A, Tensor *B, float min = 0.0f, float max = 1.0f)

.. code-block:: c++

    blablabla
    
pow
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::pow_(float)
.. doxygenfunction:: Tensor::pow(float exp)
.. doxygenfunction:: Tensor::pow(Tensor *A, Tensor *B, float exp)

.. code-block:: c++

    blablabla


powb
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::powb_(float)
.. doxygenfunction:: Tensor::powb(float base)
.. doxygenfunction:: Tensor::powb(Tensor *A, Tensor *B, float base)

.. code-block:: c++

    blablabla
    
reciprocal
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::reciprocal_()
.. doxygenfunction:: Tensor::reciprocal()
.. doxygenfunction:: Tensor::reciprocal(Tensor *A, Tensor *B)

.. code-block:: c++

.. code-block:: c++

    blablabla
    
remainder
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::remainder_(float)
.. doxygenfunction:: Tensor::remainder(float v)
.. doxygenfunction:: Tensor::remainder(Tensor *A, Tensor *B, float v)

.. code-block:: c++

    blablabla
    
    
round
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::round_()
.. doxygenfunction:: Tensor::round()
.. doxygenfunction:: Tensor::round(Tensor *A, Tensor *B)

.. code-block:: c++

    blablabla
    
rsqrt
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::rsqrt_()
.. doxygenfunction:: Tensor::rsqrt()
.. doxygenfunction:: Tensor::rsqrt(Tensor *A, Tensor *B)

.. code-block:: c++

    blablabla

Sigmoid
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sigmoid()
.. doxygenfunction:: Tensor::sigmoid(Tensor *A, Tensor *B)

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

.. doxygenfunction:: Tensor::sign(Tensor *A, Tensor *B, float zero_sign = 0.0f)

.. code-block:: c++

    Tensor* t1 = Tensor::linspace(-1,1,5);
    // [-1.00 -0.50 0.00 0.50 1.00]


    t1->sign_(5);  // In-place
    // [-1.00 -1.00 5.00 1.00 1.00]

    // Other ways
    Tensor* t2 = t1->sign(5); // returns a new tensor
    Tensor::sign(t1, t2, 5); // static
    

Sin
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sin()
.. doxygenfunction:: Tensor::sin(Tensor *A, Tensor *B)

.. code-block:: c++

    blablabla

    
Sinh
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sinh()
.. doxygenfunction:: Tensor::sinh(Tensor *A, Tensor *B)

.. code-block:: c++

    blablabla
    
Sqr
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sqr()
.. doxygenfunction:: Tensor::sqr(Tensor *A, Tensor *B)

.. code-block:: c++
    
    blablabla
    
Sqrt
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sqrt()
.. doxygenfunction:: Tensor::sqrt(Tensor *A, Tensor *B)

.. code-block:: c++

    blablabla
    
Sub
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sub(float v)
.. doxygenfunction:: Tensor::sub(Tensor *A)
.. doxygenfunction:: Tensor::sub(Tensor *A, Tensor *B, float v)
.. doxygenfunction:: Tensor::sub(Tensor *A, Tensor *B, Tensor *C)


.. code-block:: c++

    blablabla
    

    
Tan
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::tan()
.. doxygenfunction:: Tensor::tan(Tensor *A, Tensor *B)

.. code-block:: c++

    blablabla
    
Tanh
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::tanh()
.. doxygenfunction:: Tensor::tanh(Tensor *A, Tensor *B)

.. code-block:: c++

    blablabla
    
Trunc
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::trunc()
.. doxygenfunction:: Tensor::trunc(Tensor *A, Tensor *B)

.. code-block:: c++

    blablabla


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

    // Other ways
    Tensor* t3 = t1->add(t2);  // returns new tensor


div
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::div(Tensor *A, Tensor *B)

.. code-block:: c++

    blablabla

mult
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::mult(Tensor *A, Tensor *B)
.. doxygenfunction:: Tensor::mult(Tensor *A, Tensor *B, Tensor *C)

.. code-block:: c++

    blablabla

sub
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sub(Tensor *A, Tensor *B)
.. doxygenfunction:: Tensor::sub(Tensor *A, Tensor *B, Tensor *C)

.. code-block:: c++

    blablabla


Reductions
------------------

Apply lower bound
^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::maximum(float v)
.. doxygenfunction:: Tensor::maximum(Tensor *A, float v)
.. doxygenfunction:: Tensor::maximum(Tensor *A, Tensor *B, float v)

.. code-block:: c++

    blablabla


Obtain maximum values
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::maximum(Tensor *A, Tensor *B)
.. doxygenfunction:: Tensor::maximum(Tensor *A, Tensor *B, Tensor *C)

.. code-block:: c++

    blablabla


Apply upper bound
^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::minimum(float v)
.. doxygenfunction:: Tensor::minimum(Tensor *A, float v)
.. doxygenfunction:: Tensor::minimum(Tensor *A, Tensor *B, float v)

.. code-block:: c++

    blablabla


Obtain minumum values
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::minimum(Tensor *A, Tensor *B)
.. doxygenfunction:: Tensor::minimum(Tensor *A, Tensor *B, Tensor *C)

.. code-block:: c++

    blablabla


median
^^^^^^^^
.. doxygenfunction:: Tensor::median()
.. doxygenfunction:: Tensor::median(Tensor *A)


.. code-block:: c++

    blablabla


max
^^^^^^^^
.. doxygenfunction:: Tensor::max()
.. doxygenfunction:: Tensor::max(Tensor *A)
.. doxygenfunction:: Tensor::max(vector<int> axis, bool keepdims)


.. code-block:: c++

    blablabla


argmax
^^^^^^^^
.. doxygenfunction:: Tensor::argmax()
.. doxygenfunction:: Tensor::argmax(Tensor *A)
.. doxygenfunction:: Tensor::argmax(vector<int> axis, bool keepdims)


.. code-block:: c++

    blablabla


min
^^^^^^^^
.. doxygenfunction:: Tensor::min()
.. doxygenfunction:: Tensor::min(Tensor *A)
.. doxygenfunction:: Tensor::min(vector<int> axis, bool keepdims)


.. code-block:: c++

    blablabla

    
argmin
^^^^^^^^
.. doxygenfunction:: Tensor::argmin()
.. doxygenfunction:: Tensor::argmin(Tensor *A)
.. doxygenfunction:: Tensor::argmin(vector<int> axis, bool keepdims)


.. code-block:: c++

    blablabla


sum
^^^^^^^^
.. doxygenfunction:: Tensor::sum()
.. doxygenfunction:: Tensor::sum(Tensor *A)
.. doxygenfunction:: Tensor::sum(vector<int> axis, bool keepdims)


.. code-block:: c++

    blablabla


sum_abs
^^^^^^^^
.. doxygenfunction:: Tensor::sum_abs()
.. doxygenfunction:: Tensor::sum_abs(Tensor *A)
.. doxygenfunction:: Tensor::sum_abs(vector<int> axis, bool keepdims)


.. code-block:: c++

    blablabla


prod
^^^^^^^^
.. doxygenfunction:: Tensor::prod()
.. doxygenfunction:: Tensor::prod(Tensor *A)
.. doxygenfunction:: Tensor::prod(vector<int> axis, bool keepdims)


.. code-block:: c++

    blablabla


mean
^^^^^^^^
.. doxygenfunction:: Tensor::mean()
.. doxygenfunction:: Tensor::mean(Tensor *A)
.. doxygenfunction:: Tensor::mean(vector<int> axis, bool keepdims)


.. code-block:: c++

    blablabla


std
^^^^^^^^
.. doxygenfunction:: Tensor::std(bool unbiased = true)
.. doxygenfunction:: Tensor::std(Tensor *A, bool unbiased = true)
.. doxygenfunction:: Tensor::std(vector<int> axis, bool keepdims, bool unbiased = true)


.. code-block:: c++

    blablabla


var
^^^^^^^^
.. doxygenfunction:: Tensor::var(bool unbiased = true)
.. doxygenfunction:: Tensor::var(Tensor *A, bool unbiased = true)
.. doxygenfunction:: Tensor::var(vector<int> axis, bool keepdims, bool unbiased = true)


.. code-block:: c++

    blablabla


mode
^^^^^^^^
.. doxygenfunction:: Tensor::mode()
.. doxygenfunction:: Tensor::mode(Tensor *A)
.. doxygenfunction:: Tensor::mode(vector<int> axis, bool keepdims)


.. code-block:: c++

    blablabla

