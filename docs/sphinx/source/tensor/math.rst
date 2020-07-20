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

    void abs_();
    Tensor * abs();
    
acos
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::acos_
.. doxygenfunction:: Tensor::acos()
.. doxygenfunction:: Tensor::acos(Tensor *, Tensor *)


.. code-block:: c++

    void acos_();
    Tensor * acos();
    void acos(Tensor *A, Tensor *B);
    
add
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::add_(float)
.. doxygenfunction:: Tensor::add(float)
.. doxygenfunction:: Tensor::add_(Tensor *)
.. doxygenfunction:: Tensor::add(Tensor *)
.. doxygenfunction:: Tensor::add(Tensor *, Tensor *, float)
.. doxygenfunction:: Tensor::add(float, Tensor *, float, Tensor *, Tensor *, int)


.. code-block:: c++
   
    void add_(float v);
    Tensor * add(float v);
    void add_(Tensor * A);  // this = this .+ A
    Tensor * add(Tensor * A);  // this = this .+ A
    static void add(Tensor *A, Tensor *B, float v); // B = A + v
    static void add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC); // C = a*A+b*B
    
asin
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::asin_
.. doxygenfunction:: Tensor::asin()
.. doxygenfunction:: Tensor::asin(Tensor *, Tensor *)

.. code-block:: c++

    void asin_();
    Tensor * asin();
    static void asin(Tensor *A, Tensor *B);
    
atan
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::atan_()
.. doxygenfunction:: Tensor::atan()
.. doxygenfunction:: Tensor::atan(Tensor *, Tensor *)

.. code-block:: c++

    void atan_();
    Tensor * atan();
    static void atan(Tensor *A, Tensor *B);
    
ceil
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::ceil_()
.. doxygenfunction:: Tensor::ceil()
.. doxygenfunction:: Tensor::ceil(Tensor *, Tensor *)

.. code-block:: c++

    void ceil_();
    Tensor * ceil();
    static void ceil(Tensor *A, Tensor *B);


clamp
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::clamp_(float, float)
.. doxygenfunction:: Tensor::clamp(float, float)
.. doxygenfunction:: Tensor::clamp(Tensor *, Tensor *, float, float)

.. code-block:: c++

    void clamp_(float min, float max);
    Tensor * clamp(float min, float max);
    static void clamp(Tensor *A, Tensor *B, float min, float max);


    
clampmax
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::clampmax_(float)
.. doxygenfunction:: Tensor::clampmax(float)
.. doxygenfunction:: Tensor::clampmax(Tensor *, Tensor *, float)

.. code-block:: c++
   
    void clampmax_(float max);
    Tensor * clampmax(float max);
    static void clampmax(Tensor *A, Tensor *B, float max);

    
clampmin
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::clampmin_(float)
.. doxygenfunction:: Tensor::clampmin(float)
.. doxygenfunction:: Tensor::clampmin(Tensor *, Tensor *, float)

.. code-block:: c++
   
    void clampmin_(float max);
    Tensor * clampmin(float max);
    static void clampmin(Tensor *A, Tensor *B, float max);
    
cos
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cos_()
.. doxygenfunction:: Tensor::cos()
.. doxygenfunction:: Tensor::cos(Tensor *, Tensor *)

.. code-block:: c++

    void cos_();
    Tensor * cos();
    static void cos(Tensor *A, Tensor *B);
    
cosh
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cosh_()
.. doxygenfunction:: Tensor::cosh()
.. doxygenfunction:: Tensor::cosh(Tensor *, Tensor *)

.. code-block:: c++

    void cosh_();
    Tensor * cosh();
    static void cosh(Tensor *A, Tensor *B);
    
    
div
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::div_(float)
.. doxygenfunction:: Tensor::div(float)
.. doxygenfunction:: Tensor::div_(Tensor *)
.. doxygenfunction:: Tensor::div(Tensor *)
.. doxygenfunction:: Tensor::div(Tensor *, Tensor *, float)


.. code-block:: c++
   
    void div_(float v);
    Tensor * div(float v);
    void div_(Tensor * A);  // this = this ./ A
    Tensor * div(Tensor * A);  // this = this ./ A
    static void div(Tensor *A, Tensor *B, float v); // B = A / v
    

el_div
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::el_div

.. code-block:: c++

    static void el_div(Tensor *A, Tensor *B, Tensor *C, int incC);


el_mult
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::el_mult

.. code-block:: c++

    static void el_mult(Tensor *A, Tensor *B, Tensor *C, int incC);

exp
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::exp_()
.. doxygenfunction:: Tensor::exp()
.. doxygenfunction:: Tensor::exp(Tensor *, Tensor *)

.. code-block:: c++

    void exp_();
    Tensor * exp();
    static void exp(Tensor *A, Tensor *B);

floor
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::floor_()
.. doxygenfunction:: Tensor::floor()
.. doxygenfunction:: Tensor::floor(Tensor *, Tensor *)

.. code-block:: c++

    void floor_();
    Tensor * floor();
    static void floor(Tensor *A, Tensor *B);

inv
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::inv_(float)
.. doxygenfunction:: Tensor::inv(float)
.. doxygenfunction:: Tensor::inv(Tensor *, Tensor *, float)

.. code-block:: c++

    void inv_(float v=1.0f);
    Tensor * inv(float v=1.0f);
    static void inv(Tensor *A, Tensor *B, float v=1.0f);

inc
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::inc

.. code-block:: c++

    static void inc(Tensor *A, Tensor *B);
    
log
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::log_()
.. doxygenfunction:: Tensor::log()
.. doxygenfunction:: Tensor::log(Tensor *, Tensor *)

.. code-block:: c++

    void log_();
    Tensor * log();
    static void log(Tensor *A, Tensor *B);
    
log2
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::log2_()
.. doxygenfunction:: Tensor::log2()
.. doxygenfunction:: Tensor::log2(Tensor *, Tensor *)

.. code-block:: c++

    void log2_();
    Tensor * log2();
    static void log2(Tensor *A, Tensor *B);
    
log10
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::log10_()
.. doxygenfunction:: Tensor::log10()
.. doxygenfunction:: Tensor::log10(Tensor *, Tensor *)

.. code-block:: c++

    void log10_();
    Tensor * log10();
    static void log10(Tensor *A, Tensor *B);
    
logn
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logn_(float)
.. doxygenfunction:: Tensor::logn(float)
.. doxygenfunction:: Tensor::logn(Tensor *, Tensor *, float)

.. code-block:: c++

    void logn_(float n);
    Tensor * logn(float n);
    static void logn(Tensor *A, Tensor *B, float n);
    
mod
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::mod_(float)
.. doxygenfunction:: Tensor::mod(float)
.. doxygenfunction:: Tensor::mod(Tensor *, Tensor *, float)

.. code-block:: c++

    void mod_(float v);
    Tensor * mod(float v);
    static void mod(Tensor *A, Tensor *B, float v);
    
mult
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::mult_(float)
.. doxygenfunction:: Tensor::mult(float)
.. doxygenfunction:: Tensor::mult_(Tensor *)
.. doxygenfunction:: Tensor::mult(Tensor *)
.. doxygenfunction:: Tensor::mult(Tensor *, Tensor *, float)


.. code-block:: c++
   
    void mult_(float v);
    Tensor * mult(float v);
    void mult_(Tensor * A);  // this = this .* A
    Tensor * mult(Tensor * A);  // this = this .* A
    static void mult(Tensor *A, Tensor *B, float v); // B = A * v
    
neg
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::neg_()
.. doxygenfunction:: Tensor::neg()
.. doxygenfunction:: Tensor::neg(Tensor *, Tensor *)

.. code-block:: c++

    void neg_();
    Tensor * neg();
    static void neg(Tensor *A, Tensor *B);

normalize
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::normalize_(float, float)
.. doxygenfunction:: Tensor::normalize(float, float)
.. doxygenfunction:: Tensor::normalize(Tensor *, Tensor *, float, float)

.. code-block:: c++

    void normalize_(float min=0.0f, float max=1.0f);
    Tensor * normalize(float min=0.0f, float max=1.0f);
    static void normalize(Tensor *A, Tensor *B, float min=0.0f, float max=1.0f);
    
pow
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::pow_(float)
.. doxygenfunction:: Tensor::pow(float)
.. doxygenfunction:: Tensor::pow(Tensor *, Tensor *, float)

.. code-block:: c++

    void pow_(float exp);
    Tensor * pow(float exp);
    static void pow(Tensor *A, Tensor *B, float min=0.0f, float exp);


powb
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::powb_(float)
.. doxygenfunction:: Tensor::powb(float)
.. doxygenfunction:: Tensor::powb(Tensor *, Tensor *, float)

.. code-block:: c++

    void powb_(float exp);
    Tensor * powb(float exp);
    static void powb(Tensor *A, Tensor *B, float min=0.0f, float exp);
    
reciprocal
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::reciprocal_()
.. doxygenfunction:: Tensor::reciprocal()
.. doxygenfunction:: Tensor::reciprocal(Tensor *, Tensor *)

.. code-block:: c++

    void reciprocal_();
    Tensor * reciprocal();
    static void reciprocal(Tensor *A, Tensor *B);
    
remainder
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::remainder_(float)
.. doxygenfunction:: Tensor::remainder(float)
.. doxygenfunction:: Tensor::remainder(Tensor *, Tensor *, float)

.. code-block:: c++

    void remainder_(float v);
    Tensor * remainder(float v);
    static void remainder(Tensor *A, Tensor *B, float min=0.0f, float v);
    
round
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::round_()
.. doxygenfunction:: Tensor::round()
.. doxygenfunction:: Tensor::round(Tensor *, Tensor *)

.. code-block:: c++

    void round_();
    Tensor * round();
    static void round(Tensor *A, Tensor *B);
    
rsqrt
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::rsqrt_()
.. doxygenfunction:: Tensor::rsqrt()
.. doxygenfunction:: Tensor::rsqrt(Tensor *, Tensor *)

.. code-block:: c++

    void rsqrt_();
    Tensor * rsqrt();
    static void rsqrt(Tensor *A, Tensor *B);
    
sigmoid
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sigmoid_()
.. doxygenfunction:: Tensor::sigmoid()
.. doxygenfunction:: Tensor::sigmoid(Tensor *, Tensor *)

.. code-block:: c++

    void sigmoid_();
    Tensor * sigmoid();
    static void sigmoid(Tensor *A, Tensor *B);
    
sign
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sign(float)

.. doxygenfunction:: Tensor::sign(Tensor *, Tensor *, float)

.. code-block:: c++  

    static Tensor * sign(float zero_sign);
    static void sign(Tensor *A, Tensor *B, float zero_sign);
    

sin
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sin_()
.. doxygenfunction:: Tensor::sin()
.. doxygenfunction:: Tensor::sin(Tensor *, Tensor *)

.. code-block:: c++

    void sin_();
    Tensor * sin();
    static void sin(Tensor *A, Tensor *B);
    
sinh
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sinh_()
.. doxygenfunction:: Tensor::sinh()
.. doxygenfunction:: Tensor::sinh(Tensor *, Tensor *)

.. code-block:: c++

    void sinh_();
    Tensor * sinh();
    static void sinh(Tensor *A, Tensor *B);
    
sqr
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sqr_()
.. doxygenfunction:: Tensor::sqr()
.. doxygenfunction:: Tensor::sqr(Tensor *, Tensor *)

.. code-block:: c++

    void sqr_();
    Tensor * sqr();
    static void sqr(Tensor *A, Tensor *B);
    
sqrt
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sqrt_()
.. doxygenfunction:: Tensor::sqrt()
.. doxygenfunction:: Tensor::sqrt(Tensor *, Tensor *)

.. code-block:: c++

    void sqrt_();
    Tensor * sqrt();
    static void sqrt(Tensor *A, Tensor *B);
    
sub
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sub_(float)
.. doxygenfunction:: Tensor::sub(float)
.. doxygenfunction:: Tensor::sub_(Tensor *)
.. doxygenfunction:: Tensor::sub(Tensor *)
.. doxygenfunction:: Tensor::sub(Tensor *, Tensor *, float)


.. code-block:: c++
   
    void sub_(float v);
    Tensor * sub(float v);
    void sub_(Tensor * A);  // this = this .- A
    Tensor * sub(Tensor * A);  // this = this .- A
    static void sub(Tensor *A, Tensor *B, float v); // B = A - v
    

    
tan
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::tan_()
.. doxygenfunction:: Tensor::tan()
.. doxygenfunction:: Tensor::tan(Tensor *, Tensor *)

.. code-block:: c++

    void tan_();
    Tensor * tan();
    static void tan(Tensor *A, Tensor *B);
    
tanh
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::tanh_()
.. doxygenfunction:: Tensor::tanh()
.. doxygenfunction:: Tensor::tanh(Tensor *, Tensor *)

.. code-block:: c++

    void tanh_();
    Tensor * tanh();
    static void tanh(Tensor *A, Tensor *B);
    
trunc
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::trunc_()
.. doxygenfunction:: Tensor::trunc()
.. doxygenfunction:: Tensor::trunc(Tensor *, Tensor *)

.. code-block:: c++

    void trunc_();
    Tensor * trunc();
    static void trunc(Tensor *A, Tensor *B);


Binary Operations
-------------------

add
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::add(Tensor *, Tensor *)
.. doxygenfunction:: Tensor::add(Tensor *, Tensor *, Tensor *)

.. code-block:: c++

    static Tensor * add(Tensor *A, Tensor *B); // (new)C = A + B
    static void add(Tensor *A, Tensor *B, Tensor *C); // C = A + B


div
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::div(Tensor *, Tensor *)
.. doxygenfunction:: Tensor::div(Tensor *, Tensor *, Tensor *)

.. code-block:: c++

    static Tensor * div(Tensor *A, Tensor *B); // (new)C = A / B
    static void div(Tensor *A, Tensor *B, Tensor *C); // C = A / B

mult
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::mult(Tensor *, Tensor *)
.. doxygenfunction:: Tensor::mult(Tensor *, Tensor *, Tensor *)

.. code-block:: c++

    static Tensor * mult(Tensor *A, Tensor *B); // (new)C = A * B
    static void mult(Tensor *A, Tensor *B, Tensor *C); // C = A * B

sub
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sub(Tensor *, Tensor *)
.. doxygenfunction:: Tensor::sub(Tensor *, Tensor *, Tensor *)

.. code-block:: c++

    static Tensor * sub(Tensor *A, Tensor *B); // (new)C = A - B
    static void sub(Tensor *A, Tensor *B, Tensor *C); // C = A - B

Reductions
------------------

Apply lower bound
^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::maximum(float)
.. doxygenfunction:: Tensor::maximum(Tensor *, float)
.. doxygenfunction:: Tensor::maximum(Tensor *, Tensor *, float)

.. code-block:: c++
   
    Tensor * maximum(float v);
    static Tensor * maximum(Tensor * A, float v);
    static void maximum(Tensor * A, Tensor * B, float v);


Obtain maximum values
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::maximum(Tensor *, Tensor *)
.. doxygenfunction:: Tensor::maximum(Tensor *, Tensor *, Tensor *)

.. code-block:: c++
   
    static Tensor * maximum(Tensor * A, Tensor * B);
    static void maximum(Tensor * A, Tensor * B, Tensor * C);


Apply upper bound
^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::minimum(float)
.. doxygenfunction:: Tensor::minimum(Tensor *, float)
.. doxygenfunction:: Tensor::minimum(Tensor *, Tensor *, float)

.. code-block:: c++
   
    Tensor * minimum(float v);
    static Tensor * minimum(Tensor * A, float v);
    static void minimum(Tensor * A, Tensor * B, float v);


Obtain minumum values
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::minimum(Tensor *, Tensor *)
.. doxygenfunction:: Tensor::minimum(Tensor *, Tensor *, Tensor *)

.. code-block:: c++
   
    static Tensor * minimum(Tensor * A, Tensor * B);
    static void minimum(Tensor * A, Tensor * B, Tensor * C);


median
^^^^^^^^
.. doxygenfunction:: Tensor::median()
.. doxygenfunction:: Tensor::median(Tensor *)


.. code-block:: c++
   
    float median();
    static float median(Tensor * A);


max
^^^^^^^^
.. doxygenfunction:: Tensor::max()
.. doxygenfunction:: Tensor::max(Tensor *)
.. doxygenfunction:: Tensor::max(vector<int>, bool)


.. code-block:: c++
   
    float max();
    static float max(Tensor * A);
    Tensor * max(vector<int> axis, bool keepdims);


argmax
^^^^^^^^
.. doxygenfunction:: Tensor::argmax()
.. doxygenfunction:: Tensor::argmax(Tensor *)
.. doxygenfunction:: Tensor::argmax(vector<int>, bool)


.. code-block:: c++
   
    float argmax();
    static float argmax(Tensor * A);
    Tensor * argmax(vector<int> axis, bool keepdims);


min
^^^^^^^^
.. doxygenfunction:: Tensor::min()
.. doxygenfunction:: Tensor::min(Tensor *)
.. doxygenfunction:: Tensor::min(vector<int>, bool)


.. code-block:: c++
   
    float min();
    static float min(Tensor * A);
    Tensor * min(vector<int> axis, bool keepdims);

    
argmin
^^^^^^^^
.. doxygenfunction:: Tensor::argmin()
.. doxygenfunction:: Tensor::argmin(Tensor *)
.. doxygenfunction:: Tensor::argmin(vector<int>, bool)


.. code-block:: c++
   
    float argmin();
    static float argmin(Tensor * A);
    Tensor * argmin(vector<int> axis, bool keepdims);


sum
^^^^^^^^
.. doxygenfunction:: Tensor::sum()
.. doxygenfunction:: Tensor::sum(Tensor *)
.. doxygenfunction:: Tensor::sum(vector<int>, bool)


.. code-block:: c++
   
    float sum();
    static float sum(Tensor * A);
    Tensor * sum(vector<int> axis, bool keepdims);


sum_abs
^^^^^^^^
.. doxygenfunction:: Tensor::sum_abs()
.. doxygenfunction:: Tensor::sum_abs(Tensor *)
.. doxygenfunction:: Tensor::sum_abs(vector<int>, bool)


.. code-block:: c++
   
    float sum_abs();
    static float sum_abs(Tensor * A);
    Tensor * sum_abs(vector<int> axis, bool keepdims);


prod
^^^^^^^^
.. doxygenfunction:: Tensor::prod()
.. doxygenfunction:: Tensor::prod(Tensor *)
.. doxygenfunction:: Tensor::prod(vector<int>, bool)


.. code-block:: c++
   
    float prod();
    static float prod(Tensor * A);
    Tensor * prod(vector<int> axis, bool keepdims);


mean
^^^^^^^^
.. doxygenfunction:: Tensor::mean()
.. doxygenfunction:: Tensor::mean(Tensor *)
.. doxygenfunction:: Tensor::mean(vector<int>, bool)


.. code-block:: c++
   
    float mean();
    static float mean(Tensor * A);
    Tensor * mean(vector<int> axis, bool keepdims);


std
^^^^^^^^
.. doxygenfunction:: Tensor::std(bool)
.. doxygenfunction:: Tensor::std(Tensor *, bool)
.. doxygenfunction:: Tensor::std(vector<int>, bool, bool)


.. code-block:: c++
   
    float std(bool unbiased=true);
    static float std(Tensor * A, bool unbiased=true);
    Tensor * std(vector<int> axis, bool keepdims, bool unbiased=true);


var
^^^^^^^^
.. doxygenfunction:: Tensor::var(bool)
.. doxygenfunction:: Tensor::var(Tensor *, bool)
.. doxygenfunction:: Tensor::var(vector<int>, bool, bool)


.. code-block:: c++
   
    float var(bool unbiased=true);
    static float var(Tensor * A, bool unbiased=true);
    Tensor * var(vector<int> axis, bool keepdims, bool unbiased=true);


mode
^^^^^^^^
.. doxygenfunction:: Tensor::mode()
.. doxygenfunction:: Tensor::mode(Tensor *)
.. doxygenfunction:: Tensor::mode(vector<int>, bool)


.. code-block:: c++
   
    float mode();
    static float mode(Tensor * A);
    Tensor * mode(vector<int> axis, bool keepdims);


Matrix Operations
--------------------

sum
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sum2D_rowwise

.. doxygenfunction:: Tensor::sum2D_colwise

.. code-block:: c++
   
    static void sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
    static void sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);


mult
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::mult2D

.. code-block:: c++
   
    static void mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC);
