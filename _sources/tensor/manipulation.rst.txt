Manipulation
==============

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/docs/markdown/eddl_progress_tensor.md


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

.. doxygenfunction:: checkCompatibility(Tensor*, Tensor*, const string&)
.. doxygenfunction:: checkCompatibility(Tensor*, Tensor*, Tensor*, const string&)


.. code-block:: c++

    void checkCompatibility(Tensor*A, Tensor*B, const string &title);
    void checkCompatibility(Tensor*A, Tensor*B, Tensor*C, const string &title);
    


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

    vector<int> v {3,3} // Desired shape
    Tensor* tensor1 = Tensor::full(v, 10, DEV_CPU); // Creates 3x3 tensor filled with 10s
    //tensor1 => [10,10,10
    //            10,10,10
    //            10,10,10]

    bool sq = Tensor::isSquared(tensor1); //sq = true

    vector<int> v2 {3,2} // Desired shape
    Tensor* tensor2 = Tensor::full(v2, 10, DEV_CPU); // Creates 3x2 tensor filled with 10s
    //tensor1 => [10,10
    //            10,10
    //            10,10]

    bool sq2 = Tensor::isSquared(tensor2); //sq = false

Changing array shape
---------------------

reshape
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::reshape_

.. doxygenfunction:: Tensor::reshape

.. code-block:: c++

    vector<int> v {3} // Initial shape
    Tensor* tensor1 = Tensor::full(v, 10, DEV_CPU); // Creates 1D tensor filled with 10s
    //tensor1 => [10,10,10]
    
    vector<int> v2 {1,3} // Desired shape
    Tensor* tensor2 = reshape(tensor1, v2); //tensor2 has dimensions 1x3

    tensor1->reshape_(v2); //Now tensor1 has dimensions 1x3
    
    
flatten
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::flatten_
.. doxygenfunction:: Tensor::flatten

.. code-block:: c++
    
    vector<int> v1 {1,3} // Desired shape
    Tensor* tensor1 = Tensor::full(v, 10, DEV_CPU); // Creates 1x3 tensor filled with 10s
    Tensor* tensor2 = Tensor::flatten(tensor1); //tensor2 is 1D with 3 components
    
    tensor1->flatten_(); //tensor1 is now 1D with 3 components
    


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
    static Tensor* squeeze(Tensor*A);
    
unsqueeze
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::unsqueeze_
.. doxygenfunction:: Tensor::unsqueeze

.. code-block:: c++

    void unsqueeze_();
    static Tensor* unsqueeze(Tensor*A);


Joining arrays
---------------

.. doxygenfunction:: Tensor::concat

Example:

.. code-block:: c++
   :linenos:

   Tensor* t5 = Tensor::range(1, 0+3*2*2, 1.0f); t5->reshape_({3, 2, 2});
   Tensor* t6 = Tensor::range(11, 10+3*2*2, 1.0f); t6->reshape_({3, 2, 2});
   Tensor* t7 = Tensor::range(101, 100+3*2*2, 1.0f); t7->reshape_({3, 2, 2});
   Tensor* t8 = Tensor::concat({t5, t6, t7}, 2); // concat of t5, t6 and t7
    

Rearranging elements and transformations
-----------------------------------------

shift
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::shift

.. code-block:: c++
    
    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::shift(t1, t2, {50, 100}, WrappingMode::Constant, 0.0f); // Shifts t1 50 pixels in y and 100 in x.

    
rotate
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::rotate

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);
    Tensor* t3 = t2->clone();
    Tensor::rotate(t2, t3, 60.0f, {0,0}, WrappingMode::Original); //Rotates t2 60 degrees
    
scale
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::scale

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = Tensor::zeros({1, 3, 100, 100});
    Tensor::scale(t1, t2, {100, 100}); //Scale the image to 100x100 px

    
    
flip
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::flip(Tensor*, Tensor*, int)

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::flip(t1, t2, 0); // Flip along vertical axis
    
crop
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::crop

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::crop(t1, t2, {0, 250}, {200, 450}); //Crop the rectangle formed by {0,250} and {200,450}

crop_scale
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::crop_scale

.. code-block:: c++


    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::crop_scale(t1, t2, {0, 250}, {200, 450});
    
    
cutout
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cutout

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::cutout(t1, t2, {50, 100}, {100, 400});//Fill with zeros the rectangle formed by {50,100} and {100,400}

    

Tensor Data Augmentation
--------------------------

shift_random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::shift_random

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::shift_random(t1, t2, {0,50}, {10,100}); //Shifts t1 with a random shift value in y between 0 and 50 and in x between 10 and 100.
    
rotate_random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::rotate_random

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    static void rotate_random(t1, t2, {30,60}); //Rotate t1 with a random rotation factor between 30 and 60 degrees
    
scale_random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::scale_random

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::scale_random(t1, t2, {10,20}); //Scale t1 with a random scale factor between 10 and 20
    
flip_random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::flip_random

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::flip_random(t1, t2, 0); //Flip t1 on vertical axis randomly
    
crop_random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::crop_random

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::crop_random(t1, t2); //Obtain a random crop from t1
    
crop_scale_random
^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::crop_scale_random

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::crop_scale_random(t1, t2, {10,20}); //Obtain a random crop from t1 and scale it randomly with a factor between 10 and 20
    
cutout_random
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::cutout_random

.. code-block:: c++

    string fname = "../../examples/data/elephant.jpg";  // Some image
    Tensor* t1 = Tensor::load(fname);
    Tensor* t2 = new Tensor(t1->shape);

    Tensor::cutout_random(t1, t2, {50,60}, {10,100});//Set to 0 pixels in a rectangle defined by a random height between 50 and 60 and a random width between 10 and 100

Value operations
-----------------

fill
^^^^^^^
.. doxygenfunction:: Tensor::fill_(float)
.. doxygenfunction:: Tensor::fill(Tensor*, float)
.. doxygenfunction:: Tensor::fill(Tensor*, int, int, Tensor*, int, int, int)

.. code-block:: c++

    Tensor* t1 = Tensor::ones({3});
    // t1 => [1,1,1]

    Tensor::fill(t1, 50.0);
    // t1 => [50,50,50]

    Tensor* t2 = Tensor::zeros({3})
    // t2 => [0,0,0]

    Tensor::fill(t1, 0, 1, t2, 1, 2, 1);
    // t2 => [0,50,50]


    t2->fill_(3.0);
    // t2 => [3,3,3]
    
    
rand_uniform
^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::rand_uniform

.. code-block:: c++

    Tensor* t1 = Tensor::ones({3});
    t1->rand_uniform(1.0);//Fills t1 with samples from a uniform distribution.



rand_signed_uniform
^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::rand_signed_uniform

.. code-block:: c++

    Tensor* t1 = Tensor::ones({3});
    t1->rand_signed_uniform(float v);//Fills t1 with samples from a signed uniform distribution.


rand_normal
^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::rand_normal

.. code-block:: c++

    Tensor* t1 = Tensor::ones({3});
    t1->rand_normal(0, 1); //Fills t1 with samples from a normal distribution with mean 0 and std 1.
      


rand_binary
^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::rand_binary

.. code-block:: c++

    Tensor* t1 = Tensor::ones({3});
    t1->rand_binary(1);//Fills t1 with samples from a binary distribution
