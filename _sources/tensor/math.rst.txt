Mathematical functions
========================

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/eddl_progress_tensor.md


Element-wise
-------------

abs
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::abs

.. code-block:: c++

    static Tensor* abs(Tensor *A);
    
acos
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::acos

.. code-block:: c++

    static Tensor* acos(Tensor *A);
    
add
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::add(Tensor *, Tensor *)

.. doxygenfunction:: Tensor::add(float, Tensor *, float, Tensor *, Tensor *, int)

.. doxygenfunction:: Tensor::add(Tensor *, Tensor *, Tensor *)

.. code-block:: c++
   
    static Tensor* add(Tensor *A, Tensor *B);
    static void add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);
    static void add(Tensor *A, Tensor *B, Tensor *C);
    
asin
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::asin

.. code-block:: c++

    static Tensor* asin(Tensor *A);
    
atan
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::atan

.. code-block:: c++

    static Tensor* atan(Tensor *A);
    
ceil
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::ceil

.. code-block:: c++
   
    static Tensor* ceil(Tensor *A);
    
clamp
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::clamp

.. code-block:: c++
   
    static Tensor* clamp(Tensor *A, float min, float max);
    
clampmax
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::clampmax

.. code-block:: c++
   
    static Tensor* clampmax(Tensor *A, float max);
    
clampmin
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::clampmin

.. code-block:: c++

    static Tensor* clampmin(Tensor *A, float min);
    
cos
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cos

.. code-block:: c++

    static Tensor* cos(Tensor *A);
    
cosh
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cosh

.. code-block:: c++
   
    static Tensor* cosh(Tensor *A);
    
div
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::div(Tensor *, float)

.. doxygenfunction:: Tensor::div(Tensor *, Tensor *)

.. code-block:: c++  

    static Tensor* div(Tensor *A, float v);
    static Tensor* div(Tensor *A, Tensor *B);
    
exp
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::exp

.. code-block:: c++
   
    static Tensor* exp(Tensor *A);
    
floor
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::floor

.. code-block:: c++   

    static Tensor* floor(Tensor *A);
    
log
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::log

.. code-block:: c++
   
    static Tensor* log(Tensor *A);
    
log2
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::log2

.. code-block:: c++
   
    static Tensor* log2(Tensor *A);
    
log10
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::log10

.. code-block:: c++
   
    static Tensor* log10(Tensor *A);
    
logn
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::logn

.. code-block:: c++
   
    static Tensor* logn(Tensor *A, float n);
    
mod
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::mod

.. code-block:: c++
   
    static Tensor* mod(Tensor *A, float v);
    
mult
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::mult(Tensor *, Tensor *)

.. doxygenfunction:: Tensor::mult(Tensor *, float)

.. code-block:: c++
   
    static Tensor* mult(Tensor *A, float v);
     static Tensor* mult(Tensor *A, Tensor *B);
    
neg
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::neg

.. code-block:: c++  

    static Tensor* neg(Tensor *A);
    
pow
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::pow

.. code-block:: c++
   
    static Tensor* pow(Tensor *A, float exp);
    
reciprocal
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::reciprocal

.. code-block:: c++
   
    static Tensor* reciprocal(Tensor *A);
    
remainder
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::remainder

.. code-block:: c++
   
    static Tensor* remainder(Tensor *A, float v);
    
round
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::round

.. code-block:: c++
   
    static Tensor* round(Tensor *A);
    
rsqrt
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::rsqrt

.. code-block:: c++
   
    static Tensor* rsqrt(Tensor *A);
    
sigmoid
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sigmoid

.. code-block:: c++
   
    static Tensor* sigmoid(Tensor *A);
    
sign
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sign(Tensor *)

.. doxygenfunction:: Tensor::sign(Tensor *, Tensor *)

.. code-block:: c++  

    static Tensor* sign(Tensor *A);
    static void sign(Tensor *A, Tensor *B);
    
sin
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sin

.. code-block:: c++
   
    static Tensor* sin(Tensor *A);
    
sinh
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sinh

.. code-block:: c++
   
    static Tensor* sinh(Tensor *A);
    
sqr
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sqr

.. code-block:: c++
   
    static Tensor* sqr(Tensor *A);
    
sqrt
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sqrt

.. code-block:: c++

    static Tensor* sqrt(Tensor *A);
    
sub
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sub

.. code-block:: c++
   
    static Tensor* sub(Tensor *A, Tensor *B);
    
sum
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sum2D_rowwise

.. doxygenfunction:: Tensor::sum2D_colwise

.. doxygenfunction:: Tensor::sum_abs

.. code-block:: c++
   
    static void sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
    static void sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);
    static Tensor* sum_abs(Tensor *A);
    
tan
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::tan

.. code-block:: c++


    static Tensor* tan(Tensor *A);
    
tanh
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::tanh

.. code-block:: c++
   
    static Tensor* tanh(Tensor *A);
    
trunc
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::trunc

.. code-block:: c++
   
    static Tensor* trunc(Tensor *A);




Reductions
------------

max
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::max

.. code-block:: c++
   
    float max();
    
min
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::min

.. code-block:: c++

    float min();
