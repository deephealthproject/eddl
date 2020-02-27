Mathematical functions
========================

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/eddl_progress_tensor.md


Element-wise
-------------

abs
^^^^^^^^^^^^

.. code-block:: c++

    static Tensor* abs(Tensor *A);
    
acos
^^^^^^^^^^^^

.. code-block:: c++

    static Tensor* acos(Tensor *A);
    
add
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* add(Tensor *A, Tensor *B);
    static void add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);
    static void add(Tensor *A, Tensor *B, Tensor *C);
    
asin
^^^^^^^^^^^^

.. code-block:: c++

    static Tensor* asin(Tensor *A);
    
atan
^^^^^^^^^^^^

.. code-block:: c++

    static Tensor* atan(Tensor *A);
    
ceil
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* ceil(Tensor *A);
    
clamp
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* clamp(Tensor *A, float min, float max);
    
clampmax
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* clampmax(Tensor *A, float max);
    
clampmin
^^^^^^^^^^^^

.. code-block:: c++

    static Tensor* clampmin(Tensor *A, float min);
    
cos
^^^^^^^^^^^^

.. code-block:: c++

    static Tensor* cos(Tensor *A);
    
cosh
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* cosh(Tensor *A);
    
div
^^^^^^^^^^^^

.. code-block:: c++  

    static Tensor* div(Tensor *A, float v);
    static Tensor* div(Tensor *A, Tensor *B);
    
exp
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* exp(Tensor *A);
    
floor
^^^^^^^^^^^^

.. code-block:: c++   

    static Tensor* floor(Tensor *A);
    
log
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* log(Tensor *A);
    
log2
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* log2(Tensor *A);
    
log10
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* log10(Tensor *A);
    
logn
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* logn(Tensor *A, float n);
    
mod
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* mod(Tensor *A, float v);
    
mult
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* mult(Tensor *A, float v);
     static Tensor* mult(Tensor *A, Tensor *B);
    
neg
^^^^^^^^^^^^

.. code-block:: c++  

    static Tensor* neg(Tensor *A);
    
pow
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* pow(Tensor *A, float exp);
    
reciprocal
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* reciprocal(Tensor *A);
    
remainder
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* remainder(Tensor *A, float v);
    
round
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* round(Tensor *A);
    
rsqrt
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* rsqrt(Tensor *A);
    
sigmoid
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* sigmoid(Tensor *A);
    
sign
^^^^^^^^^^^^

.. code-block:: c++  

    static Tensor* sign(Tensor *A);
    static void sign(Tensor *A, Tensor *B);
    
sin
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* sin(Tensor *A);
    
sinh
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* sinh(Tensor *A);
    
sqr
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* sqr(Tensor *A);
    
sqrt
^^^^^^^^^^^^

.. code-block:: c++

    static Tensor* sqrt(Tensor *A);
    
sub
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* sub(Tensor *A, Tensor *B);
    
sum
^^^^^^^^^^^^

.. code-block:: c++
   
    static void sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
    static void sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);
    static Tensor* sum_abs(Tensor *A);
    
tan
^^^^^^^^^^^^

.. code-block:: c++


    static Tensor* tan(Tensor *A);
    
tanh
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* tanh(Tensor *A);
    
trunc
^^^^^^^^^^^^

.. code-block:: c++
   
    static Tensor* trunc(Tensor *A);




Reductions
------------

max
^^^^^^^^^^^^

.. code-block:: c++
   
    float max();
    
min
^^^^^^^^^^^^

.. code-block:: c++

    float min();
