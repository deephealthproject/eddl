Manipulation
==============

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/eddl_progress_tensor.md


Devices and information
--------------------------

Move to CPU
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::toCPU

.. code-block:: c++

    Tensor* tensor1 = Tensor::logspace(0.1, 1.0, 5, 10.0, DEV_GPU); //tensor1 is in GPU
    Tensor::toCPU(tensor1);
    // tensor1 is now in CPU

Move to GPU
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::toGPU

.. code-block:: c++

    Tensor* tensor1 = Tensor::logspace(0.1, 1.0, 5, 10.0, DEV_CPU); //tensor1 is in CPU
    Tensor::toGPU(tensor1);
    // tensor1 is now in GPU


Check tensor device
^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isCPU
.. doxygenfunction:: Tensor::isGPU
.. doxygenfunction:: Tensor::isFPGA
.. doxygenfunction:: Tensor::getDeviceName

.. code-block:: c++

    Tensor* tensor1 = Tensor::logspace(0.1, 1.0, 5, 10.0, DEV_GPU); //tensor1 is in GPU
    tensor1->isCPU(); // returns 0
    tensor1->isGPU(); // returns 1
    tensor1->isFPGA(); // returns 0
    tensor1->getDeviceName(); // returns GPU


Check compatibility
^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: checkCompatibility(Tensor *, Tensor *, const string&)
.. doxygenfunction:: checkCompatibility(Tensor *, Tensor *, Tensor *, const string&)


.. code-block:: c++

    void checkCompatibility(Tensor *A, Tensor *B, const string &title);
    void checkCompatibility(Tensor *A, Tensor *B, Tensor *C, const string &title);
    


Get information from tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::info

.. code-block:: c++

    Tensor* tensor1 = Tensor::logspace(0.1, 1.0, 5, 10.0, DEV_GPU); //tensor1 is in GPU
    tensor1->info(); // prints on standard output the information of tensor1


Print tensor contents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::print

.. code-block:: c++

    Tensor* t = Tensor::randn({3, 3});
    t->print();

        [
        [-1.106357 0.176572 -0.148911]
        [0.989854 -1.420635 -0.334201]
        [-0.647039 0.876878 -0.305620]
        ]

.. code-block:: c++

    Tensor* t = Tensor::randn({3, 3});
    t->print(1);

        [
        [-1.1 0.2 -0.1]
        [1.0 -1.4 -0.3]
        [-0.6 0.9 -0.3]
        ]

.. code-block:: c++

    Tensor* t = Tensor::randn({3, 3});
    t->print(0, true);

        [
        -1 0 -0 1 -1 -0 -1 1 -0
        ]


Dimension check
^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::isSquared

.. code-block:: c++

    bool isSquared(Tensor* A);

Changing array shape
---------------------

reshape
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::reshape_

.. doxygenfunction:: Tensor::reshape

.. code-block:: c++

    void reshape_(const vector<int> &new_shape);
    static Tensor* reshape(Tensor *A, const vector<int> &shape);
    
flatten
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::flatten_
.. doxygenfunction:: Tensor::flatten

.. code-block:: c++

    void flatten_();
    static Tensor* flatten(Tensor *A);


resize
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::resize

.. code-block:: c++

    void resize(int b, float *fptr=nullptr);


Transpose-like operations
--------------------------


permute
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::permute_
.. doxygenfunction:: Tensor::permute

.. code-block:: c++

    void permute_(const vector<int>& dims);
    static Tensor* permute(Tensor* t, const vector<int>& dims);
    
moveaxis
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::moveaxis

.. code-block:: c++

    static Tensor* moveaxis(Tensor* t, int source, int destination);
    
swapaxis
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::swapaxis

.. code-block:: c++

    static Tensor* swapaxis(Tensor* t, int axis1, int axis2);


Changing number of dimensions
-------------------------------

squeeze
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::squeeze_
.. doxygenfunction:: Tensor::squeeze

.. code-block:: c++

    void squeeze_();
    static Tensor* squeeze(Tensor *A);
    
unsqueeze
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::unsqueeze_
.. doxygenfunction:: Tensor::unsqueeze

.. code-block:: c++

    void unsqueeze_();
    static Tensor* unsqueeze(Tensor *A);


Joining arrays
---------------

.. doxygenfunction:: Tensor::concat

Example:

.. code-block:: c++
   :linenos:

    static Tensor* concat(const vector<Tensor*> t, unsigned int axis=0, Tensor* output=nullptr);
    

Rearranging elements and transformations
-----------------------------------------

shift
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::shift

.. code-block:: c++

    static void shift(Tensor *A,Tensor *B, vector<int> shift, string mode="constant", float constant=0.0f);
    
rotate
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::rotate

.. code-block:: c++

    static void rotate(Tensor *A, Tensor *B, float angle, vector<int> offset_center={0,0}, string mode="constant", float constant=0.0f);
    
scale
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::scale

.. code-block:: c++

    static void scale(Tensor *A, Tensor *B, vector<int> new_shape, string mode="nearest", float constant=0.0f);
    
flip
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::flip(Tensor *, Tensor *, int)

.. code-block:: c++

    static void flip(Tensor *A, Tensor *B, int axis=0);
    
crop
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::crop

.. code-block:: c++

    static void crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant=0.0f);
    
crop_scale
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::crop_scale

.. code-block:: c++

    static void crop_scale(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, string mode="nearest", float constant=0.0f);
    
cutout
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cutout

.. code-block:: c++

    static void cutout(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant=0.0f);
    

Tensor Data Augmentation
--------------------------

shift_random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::shift_random

.. code-block:: c++

    static void shift_random(Tensor *A,Tensor *B, vector<float> factor_x, vector<float> factor_y, string mode="constant", float constant=0.0f);
    
rotate_random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::rotate_random

.. code-block:: c++

    static void rotate_random(Tensor *A, Tensor *B, vector<float> factor, vector<int> offset_center={0,0}, string mode="constant", float constant=0.0f);
    
scale_random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::scale_random

.. code-block:: c++

    static void scale_random(Tensor *A, Tensor *B, vector<float> factor, string mode="nearest", float constant=0.0f);
    
flip_random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::flip_random

.. code-block:: c++

    static void flip_random(Tensor *A, Tensor *B, int axis);
    
crop_random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::crop_random

.. code-block:: c++

    static void crop_random(Tensor *A, Tensor *B);
    
crop_scale_random
^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::crop_scale_random

.. code-block:: c++

    static void crop_scale_random(Tensor *A, Tensor *B, vector<float> factor, string mode="nearest", float constant=0.0f);
    
cutout_random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cutout_random

.. code-block:: c++

    static void cutout_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant=0.0f);

Value operations
-----------------

fill
^^^^^^^
.. doxygenfunction:: Tensor::fill_(float)
.. doxygenfunction:: Tensor::fill(Tensor *, float)
.. doxygenfunction:: Tensor::fill(Tensor *, int, int, Tensor *, int, int, int)

.. code-block:: c++

    void fill_(float v);
    static void fill(Tensor* A, float v);
    static void fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc);

rand_uniform
^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::rand_uniform

.. code-block:: c++

    void rand_uniform(float v);



rand_signed_uniform
^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::rand_signed_uniform

.. code-block:: c++

    void rand_signed_uniform(float v);


rand_normal
^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::rand_normal

.. code-block:: c++

    void rand_normal(float m, float s, bool fast_math=true);
      


rand_binary
^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::rand_binary

.. code-block:: c++

    void rand_binary(float v);