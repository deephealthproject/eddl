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

    Tensor* toCPU(Tensor *A);

Move to GPU
^^^^^^^^^^^^

.. doxygenfunction:: Tensor::toGPU

.. code-block:: c++

    Tensor* toGPU(Tensor *A);


Check tensor device
^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::isCPU
.. doxygenfunction:: Tensor::isGPU
.. doxygenfunction:: Tensor::isFPGA
.. doxygenfunction:: Tensor::getDeviceName

.. code-block:: c++

    int isCPU();
    int isGPU();
    int isFPGA();
    string getDeviceName();

Get information from tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::info

.. code-block:: c++

    void info();


Print tensor contents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: Tensor::print

.. code-block:: c++

    void info();


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

.. doxygenfunction:: Tensor::flatten

.. code-block:: c++

    static Tensor* flatten(Tensor *A);


Transpose-like operations
--------------------------


permute
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::permute

.. code-block:: c++

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

.. doxygenfunction:: Tensor::squeeze

.. code-block:: c++

    static Tensor* squeeze(Tensor *A);
    
unsqueeze
^^^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::unsqueeze

.. code-block:: c++

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


   