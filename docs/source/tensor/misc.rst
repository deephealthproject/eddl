Miscellaneous
==============

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/eddl_progress_tensor.md


Functions
------------

toCPU
^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::toCPU

.. code-block:: c++

    void toCPU(int dev=DEV_CPU);

toGPU
^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::toGPU

.. code-block:: c++

    void toGPU(int dev=DEV_GPU);


isCPU
^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isCPU

.. code-block:: c++

    int isCPU();


isGPU
^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isGPU

.. code-block:: c++

    int isGPU();


isFPGA
^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isFPGA

.. code-block:: c++

    int isFPGA();


isSquared
^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isSquared

.. code-block:: c++

    static bool isSquared(Tensor *A);


copy
^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::copy

.. code-block:: c++

    static void copy(Tensor *A, Tensor *B);


clone
^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::clone

.. code-block:: c++

    Tensor* clone();


info
^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::info

.. code-block:: c++

    void info();


print
^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::print

.. code-block:: c++

    void print(bool asInt=false, bool raw=false);


valid_indices
^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::valid_indices

.. code-block:: c++

    bool valid_indices(vector<int> indices);


get_address_rowmajor
^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::get_address_rowmajor

.. code-block:: c++

    int get_address_rowmajor(vector<int> indices);
