Manipulation
==============

Devices and information
--------------------------

Move to CPU
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::toCPU

.. code-block:: c++

    Tensor* t1 = Tensor::zeros({2, 3}, DEV_GPU); // Tensor created in the GPU
    t1->toCPU(); // Tensor transferred to CPU


Move to GPU
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::toGPU

.. code-block:: c++

    Tensor* t1 = Tensor::zeros({2, 3}, DEV_CPU); // Tensor created in the CPU
    t1->toGPU(); // Tensor transferred to GPU


Check tensor device
^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isCPU

.. code-block:: c++

    Tensor* t1 = Tensor::zeros({2, 3}, DEV_GPU); // Tensor created in the GPU
    t1->isCPU(); // returns 0


.. doxygenfunction:: Tensor::isGPU

.. code-block:: c++

    Tensor* t1 = Tensor::zeros({2, 3}, DEV_GPU); // Tensor created in the GPU
    t1->isGPU(); // returns 1


.. doxygenfunction:: Tensor::isFPGA

.. code-block:: c++

    Tensor* t1 = Tensor::zeros({2, 3}, DEV_GPU); // Tensor created in the GPU
    t1->isFPGA(); // returns 0


.. doxygenfunction:: Tensor::getDeviceName

.. code-block:: c++

    Tensor* t1 = Tensor::zeros({2, 3}, DEV_GPU); // Tensor created in the GPU
    t1->getDeviceName(); // returns "GPU"


Get information from tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::info

.. code-block:: c++

   Tensor* t1 = Tensor::empty({2, 3});
   t1->info();
   // -------------------------------
   // class:         Tensor
   // ndim:          2
   // shape:         (2, 3)
   // strides:       (3, 1)
   // itemsize:      6
   // contiguous:    1
   // order:         C
   // data pointer:  0x7f827a6060d8
   // is shared:     0
   // type:          float (4 bytes)
   // device:        CPU (code = 0)
   // -------------------------------



Print tensor contents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::print

.. code-block:: c++

   Tensor* t1 = Tensor::randn({3, 3});
   t1->print();
   // [
   // [0.135579 -0.208483 0.537894]
   // [0.666481 0.242867 -1.957334]
   // [-1.447633 1.231033 0.670430]
   // ]

   t1->print(0);  // No decimals
   // [
   // [0 -0 1]
   // [1 0 -2]
   // [-1 1 1]
   // ]

   t1->print(3);   // 3 decimals
   // [
   // [0.136 -0.208 0.538]
   // [0.666 0.243 -1.957]
   // [-1.448 1.231 0.670]
   // ]

   t1->print(3, true);   // 3 decimals, presented in row major
   // [
   // 0.136 -0.208 0.538 0.666 0.243 -1.957 -1.448 1.231 0.670
   // ]


Dimension check
^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isSquared

.. code-block:: c++

    Tensor* t1 = Tensor::zeros({3, 3});
    Tensor::isSquared(t1);
    // true

    Tensor* t2 = Tensor::zeros({3, 3, 3});
    Tensor::isSquared(t2);
    // true

    Tensor* t3 = Tensor::zeros({3, 1, 3});
    Tensor::isSquared(t3);
    // false


Changing array shape
---------------------

reshape
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::reshape(const vector<int> &new_shape)

.. code-block:: c++


    Tensor* t1 = Tensor::zeros({3, 4});
    // [
    // [0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00]
    // [0.00 0.00 0.00 0.00]
    // ]

    t1->reshape_({6, 2});
    // [
    // [0.00 0.00]
    // [0.00 0.00]
    // [0.00 0.00]
    // [0.00 0.00]
    // [0.00 0.00]
    // [0.00 0.00]
    // ]

    // Other ways
    Tensor* t2 = Tensor::reshape(t1, {6, 2}); // returns new tensor
    t1->reshape(t2, {6,2}) // static


flatten
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::flatten()

.. code-block:: c++
    
    Tensor* t1 = Tensor::zeros({3, 4});
   // [
   // [0.00 0.00 0.00 0.00]
   // [0.00 0.00 0.00 0.00]
   // [0.00 0.00 0.00 0.00]
   // ]

   t1->flatten_();  // In-place
   // [0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00]

   // Other ways

   Tensor *t2 = Tensor::flatten(t1);  // returns new tensor
   t1->flatten(t2) // static
   

Tiling arrays
--------------------------

repeat
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::repeat(Tensor* A, const vector<unsigned int>& repeats, unsigned int axis=0, Tensor* output=nullptr, bool derivative=false)
.. doxygenfunction:: Tensor::repeat(Tensor* A, unsigned int repeats, unsigned int axis=0, Tensor* output=nullptr, bool derivative=false);

.. code-block:: c++

    Tensor* t1 = Tensor::range(1, 6); t1->reshape_({2, 3});
    // [
    // [1 2 3]
    // [4 5 6]
    // ]

    // Repeat all rows 2 times. (repeat=2, axis=0)
    Tensor *t2 = Tensor::repeat(t1, 2, 0);  // returns new tensor
    // [
    // [1 2 3]
    // [1 2 3]
    // [4 5 6]
    // [4 5 6]
    // ]


    // Repeat col 1 => 3 times; col 2 => 2 times; col 3 => 1 time. (repeat=[3,2,1], axis=1)
    Tensor *t2 = Tensor::repeat(t1, {3, 2, 1}, 1);  // returns new tensor
    // [
    // [1 1 1 2 2 3]
    // [4 4 4 5 5 6]
    // ]



tile
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::tile(Tensor* A, const vector<int>& repeats)

.. code-block:: c++

    // New tensor
    Tensor* t1 = new Tensor({1, 2}, {2, 1});
    // [
    // [1]
    // [2]
    // ]

    // Repeat all rows and columns 2 times each
    Tensor* t1_res = Tensor::tile(t1, {2, 2});
    // [
    // [1 1]
    // [2 2]
    // [1 1]
    // [2 2]
    // ]


    // New tensor
    Tensor* t2 = new Tensor({1, 2, 3}, {3, 1});
    // [
    // [1]
    // [2]
    // [3]
    // ]

    // Repeat columns three times but not rows
    Tensor* t2_res = Tensor::tile(t2, {1, 3});
    // [
    // [1 1 1]
    // [2 2 2]
    // [3 3 3]
    // ]


broadcast
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::broadcast(Tensor* A, Tensor* B);

.. code-block:: c++

        // Example: Image - constant RGB

        // Define mean
        auto* mean = new Tensor( {0.485, 0.456, 0.406}, {3}, DEV_CPU);
        // [0.485 0.456 0.406]

        // Fake image
        auto* image = Tensor::ones( {3, 224, 244});
        // -------------------------------
        // class:         Tensor
        // ndim:          3
        // shape:         (3, 224, 244)
        // strides:       (54656, 244, 1)
        // itemsize:      163968
        // contiguous:    1
        // order:         C
        // data pointer:  0x56305561baa8
        // is shared:     0
        // type:          float (4 bytes)
        // device:        CPU (code = 0)
        // -------------------------------

        // Compute broadcast for mean
        Tensor* mean_broadcasted = Tensor::broadcast(mean, image);
        // -------------------------------
        // class:         Tensor
        // ndim:          3
        // shape:         (3, 224, 244)
        // strides:       (54656, 244, 1)
        // itemsize:      163968
        // contiguous:    1
        // order:         C
        // data pointer:  0x56305561f2c8
        // is shared:     0
        // type:          float (4 bytes)
        // device:        CPU (code = 0)
        // -------------------------------

        // Apply: X-mean
        image->sub_(mean_broadcasted);


Transpose-like operations
--------------------------


permute
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::permute(const vector<int> &dims)

.. code-block:: c++

   Tensor* t1 = Tensor::range(1, 24); t1->reshape_({2, 3, 4});
   // [
   // [[1.00 2.00 3.00 4.00]     [5.00 6.00 7.00 8.00]     [9.00 10.00 11.00 12.00]]
   // [[13.00 14.00 15.00 16.00] [17.00 18.00 19.00 20.00] [21.00 22.00 23.00 24.00]]
   // ]

   t1->permute_({2, 1, 0});  // In-place
   // [
   // [[1.00 13.00] [5.00 17.00] [9.00 21.00]]
   // [[2.00 14.00] [6.00 18.00] [10.00 22.00]]
   // [[3.00 15.00] [7.00 19.00] [11.00 23.00]]
   // [[4.00 16.00] [8.00 20.00] [12.00 24.00]]
   // ]

   // Other ways
   Tensor *t2 = Tensor::permute(t1, {2, 1, 0});  // returns new tensor
   t1->permute(t2, {2, 1, 0});  // static
   


moveaxis
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::moveaxis(int source, int destination)

.. code-block:: c++

   Tensor* t1 = Tensor::zeros({1, 2, 3, 4});
   // ndim:          4
   // shape:         (1, 2, 3, 4)
   // strides:       (24, 12, 4, 1)

   t1->moveaxis_(0, 2);  // In-place
   // ndim:          4
   // shape:         (2, 3, 1, 4)
   // strides:       (12, 4, 4, 1)

   // Other ways
   Tensor *t2 = Tensor::moveaxis(t1, 0, 2);  // returns new tensor
   t1->moveaxis(t2, 0, 2);  // static


swapaxis
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::swapaxis(int axis1, int axis2)

.. code-block:: c++

   Tensor* t1 = Tensor::zeros({1, 2, 3, 4});
   // ndim:          4
   // shape:         (1, 2, 3, 4)
   // strides:       (24, 12, 4, 1)

   t1->swapaxis_(0, 2);  // In-place
   // ndim:          4
   // shape:         (3, 2, 1, 4)
   // strides:       (8, 4, 4, 1)

   // Other ways
   Tensor *t2 = Tensor::swapaxis(t1, 0, 2);  // returns new tensor
   t1->swapaxis(t2, 0, 2); // static
   


Changing number of dimensions
-------------------------------

squeeze
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::squeeze(Tensor *A, int axis = -1)

.. code-block:: c++

   Tensor* t1 = Tensor::zeros({1, 3, 4, 1});
   // ndim:          4
   // shape:         (1, 3, 4, 1)
   // strides:       (12, 4, 1, 1)

   Tensor* t2 = t1->squeeze();  // returns new tensor
   // shape:         (3, 4)

   Tensor* t3 = t1->squeeze(0);  // returns new tensor
   // shape:         (3, 4, 1)

   Tensor* t3 = t1->squeeze(3);  // returns new tensor
   // shape:         (1, 3, 4)

   // Other ways
   t1->squeeze_(); // In-place
   Tensor *t2 = Tensor::squeeze(t1);  // static



unsqueeze
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::unsqueeze(Tensor *A, int axis = 0)

.. code-block:: c++

   Tensor* t1 = Tensor::zeros({2, 3, 4});
   // ndim:          3
   // shape:         (2, 3, 4)
   // strides:       (12, 4, 1)

   Tensor* t2 = t1->unsqueeze(); // returns new tensor
    // shape:         (1, 2, 3, 4)

   Tensor* t3 = t1->unsqueeze(2); // returns new tensor
    // shape:         (2, 3, 4, 1)

   // Other ways
   t1->unsqueeze_();  // In-place
   Tensor *t2 = Tensor::unsqueeze(t1);  // Static


Joining arrays
---------------

concatenate
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::concat

Example:

.. code-block:: c++

   Tensor* t1 = Tensor::full({2, 2, 2}, 2);
   Tensor* t2 = Tensor::full({2, 2, 2}, 5);

   Tensor* t3 = Tensor::concat({t1, t2});
   // ndim:          3
   // shape:         (4, 2, 2)
   // strides:       (4, 2, 1)

   // [
   // [[2.00 2.00] [2.00 2.00]]
   // [[2.00 2.00] [2.00 2.00]]
   // [[5.00 5.00] [5.00 5.00]]
   // [[5.00 5.00] [5.00 5.00]]
   // ]

   Tensor *t4 = Tensor::concat({t1, t2}, 2);

   // ndim:          3
   // shape:         (2, 2, 4)
   // strides:       (8, 4, 1)
   //
   // [
   // [[2.00 2.00 5.00 5.00] [2.00 2.00 5.00 5.00]]
   // [[2.00 2.00 5.00 5.00] [2.00 2.00 5.00 5.00]]
   // ]


stack
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::stack

Example:

.. code-block:: c++

    Tensor* t1 = Tensor::full({2, 2}, 2);
    Tensor* t2 = Tensor::full({2, 2}, 5);

    Tensor* t3 = Tensor::stack({t1, t2});  // axis = 0
    // [
    // [[2.00 2.00] [2.00 2.00]]
    // [[5.00 5.00] [5.00 5.00]]
    // ]

    Tensor *t4 = Tensor::stack({t1, t2}, 1);  // axis = 1
    // [
    // [[2.00 2.00] [5.00 5.00]]
    // [[2.00 2.00] [5.00 5.00]]
    // ]


Value operations
-----------------

Fill constant
^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::fill(Tensor *A, float v)

.. code-block:: c++

    Tensor* t1 = Tensor::empty({2, 3});

    t1->fill_(3.0f);  // In-place
    // [
    // [3.00 3.00 3.00]
    // [3.00 3.00 3.00]
    // ]

    // Other ways
    Tensor* t2 = t1->fill(3.0f); // returns new tensor
    Tensor::fill(t1, 3.0f);  // static
    

    
    
    
Fill Random Uniform
^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::fill_rand_uniform_

.. code-block:: c++

    Tensor* t1 = Tensor::empty({2, 3});

    t1->fill_rand_uniform_(1.0f);  // In-place
    // [
    // [0.10 0.53 0.88]
    // [0.57 0.57 0.89]
    // ]

    // Other ways
    Tensor* t2 = t1->fill_rand_uniform(1.0f); // returns new tensor
    Tensor::fill_rand_uniform(t1, 1.0f);  // static

    



Fill Random Signed Uniform
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::fill_rand_signed_uniform_

.. code-block:: c++

    Tensor* t1 = Tensor::empty({2, 3});

    t1->fill_rand_signed_uniform_(1.0f);  // In-place
    // [
    // [0.22 -0.34 -0.78]
    // [-0.03 0.10 0.90]
    // ]

    // Other ways
    Tensor* t2 = t1->fill_rand_signed_uniform(1.0f); // returns new tensor
    Tensor::fill_rand_signed_uniform(t1, 1.0f);  // static
    


Fill Random Normal
^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::fill_rand_normal_

.. code-block:: c++

    Tensor* t1 = Tensor::empty({2, 3});

    t1->fill_rand_normal_(0.0f, 1.0f);  // In-place
    // [
    // [-0.57 0.49 -1.09]
    // [0.75 0.37 -0.32]
    // ]

    // Other ways
    Tensor* t2 = t1->fill_rand_normal(0.0f, 1.0f); // returns new tensor
    Tensor::fill_rand_normal(t1, 0.0f, 1.0f);  // static
    


Fill Random Binary
^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::fill_rand_binary_

.. code-block:: c++

    Tensor* t1 = Tensor::empty({2, 3});

    t1->fill_rand_binary_(0.5f);  // In-place
    // [
    // [0.00 1.00 0.00]
    // [1.00 1.00 0.00]
    // ]

    // Other ways
    Tensor* t2 = t1->fill_rand_binary(0.5f); // returns new tensor
    Tensor::fill_rand_binary(t1, 0.5f);  // static


