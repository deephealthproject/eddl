Mathematical functions
========================

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/eddl_progress_tensor.md


Element-wise
-------------

abs
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::abs

.. code-block:: c++

    static Tensor* abs(Tensor *A);
    
acos
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::acos

.. code-block:: c++

    static Tensor* acos(Tensor *A);
    
add
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::add(Tensor *, float)
.. doxygenfunction:: eddlT::add(Tensor *, Tensor *)

###Inplace versions

.. doxygenfunction:: eddlT::add_(Tensor *, float)
.. doxygenfunction:: eddlT::add_(Tensor *, Tensor *)

.. doxygenfunction:: Tensor::add(float, Tensor *, float, Tensor *, Tensor *, int)

.. doxygenfunction:: Tensor::add(Tensor *, Tensor *, Tensor *)

.. code-block:: c++
   
    static Tensor* add(Tensor *A, Tensor *B);
    static void add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);
    static void add(Tensor *A, Tensor *B, Tensor *C);
    
asin
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::asin

.. code-block:: c++

    static Tensor* asin(Tensor *A);
    
atan
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::atan

.. code-block:: c++

    static Tensor* atan(Tensor *A);
    
ceil
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::ceil

.. code-block:: c++
   
    static Tensor* ceil(Tensor *A);
    
clamp
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::clamp

.. code-block:: c++
   
    static Tensor* clamp(Tensor *A, float min, float max);
    
clampmax
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::clampmax

.. code-block:: c++
   
    static Tensor* clampmax(Tensor *A, float max);
    
clampmin
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::clampmin

.. code-block:: c++

    static Tensor* clampmin(Tensor *A, float min);
    
cos
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::cos

.. code-block:: c++

    static Tensor* cos(Tensor *A);
    
cosh
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::cosh

.. code-block:: c++
   
    static Tensor* cosh(Tensor *A);
    
div
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::div(Tensor *, float)

.. doxygenfunction:: eddlT::div(Tensor *, Tensor *)

.. code-block:: c++  

    static Tensor* div(Tensor *A, float v);
    static Tensor* div(Tensor *A, Tensor *B);
    
exp
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::exp

.. code-block:: c++
   
    static Tensor* exp(Tensor *A);
    
floor
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::floor

.. code-block:: c++   

    static Tensor* floor(Tensor *A);
    
log
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::log

.. code-block:: c++
   
    static Tensor* log(Tensor *A);
    
log2
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::log2

.. code-block:: c++
   
    static Tensor* log2(Tensor *A);
    
log10
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::log10

.. code-block:: c++
   
    static Tensor* log10(Tensor *A);
    
logn
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::logn

.. code-block:: c++
   
    static Tensor* logn(Tensor *A, float n);
    
mod
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::mod

.. code-block:: c++
   
    static Tensor* mod(Tensor *A, float v);
    
mult
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::mult(Tensor *, Tensor *)

.. doxygenfunction:: eddlT::mult(Tensor *, float)

.. code-block:: c++
   
    static Tensor* mult(Tensor *A, float v);
     static Tensor* mult(Tensor *A, Tensor *B);
    
neg
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::neg

.. code-block:: c++  

    static Tensor* neg(Tensor *A);
    
pow
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::pow

.. code-block:: c++
   
    static Tensor* pow(Tensor *A, float exp);
    
reciprocal
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::reciprocal

.. code-block:: c++
   
    static Tensor* reciprocal(Tensor *A);
    
remainder
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::remainder

.. code-block:: c++
   
    static Tensor* remainder(Tensor *A, float v);
    
round
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::round

.. code-block:: c++
   
    static Tensor* round(Tensor *A);
    
rsqrt
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::rsqrt

.. code-block:: c++
   
    static Tensor* rsqrt(Tensor *A);
    
sigmoid
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::sigmoid

.. code-block:: c++
   
    static Tensor* sigmoid(Tensor *A);
    
sign
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::sign(Tensor *)

.. doxygenfunction:: eddlT::sign(Tensor *, Tensor *)

.. code-block:: c++  

    static Tensor* sign(Tensor *A);
    static void sign(Tensor *A, Tensor *B);
    
sin
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::sin

.. code-block:: c++
   
    static Tensor* sin(Tensor *A);
    
sinh
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::sinh

.. code-block:: c++
   
    static Tensor* sinh(Tensor *A);
    
sqr
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::sqr

.. code-block:: c++
   
    static Tensor* sqr(Tensor *A);
    
sqrt
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::sqrt

.. code-block:: c++

    static Tensor* sqrt(Tensor *A);
    
sub
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::sub(Tensor *,float )
.. doxygenfunction:: eddlT::sub(Tensor *, Tensor *)

###Inplace versions

.. doxygenfunction:: eddlT::sub_(Tensor *, float)
.. doxygenfunction:: eddlT::sub_(Tensor *, Tensor *)

.. code-block:: c++
   
    static Tensor* sub(Tensor *A, Tensor *B);
    
sum
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::sum2D_rowwise

.. doxygenfunction:: Tensor::sum2D_colwise

.. doxygenfunction:: eddlT::sum_abs

.. code-block:: c++
   
    static void sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
    static void sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);
    static Tensor* sum_abs(Tensor *A);
    
tan
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::tan

.. code-block:: c++


    static Tensor* tan(Tensor *A);
    
tanh
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::tanh

.. code-block:: c++
   
    static Tensor* tanh(Tensor *A);
    
trunc
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::trunc

.. code-block:: c++
   
    static Tensor* trunc(Tensor *A);




Reductions
------------

max
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::max

.. code-block:: c++
   
    float max();
    
min
^^^^^^^^^^^^

.. doxygenfunction:: eddlT::min

.. code-block:: c++

    float min();
