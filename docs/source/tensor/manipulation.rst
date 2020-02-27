Manipulation
==============

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/eddl_progress_tensor.md


Constructor
------------

Create an uninitialized tensor

Example:

.. code-block:: c++
   :linenos:

    Tensor(const vector<int> &shape, float *fptr, int dev=DEV_CPU);


Changing array shape
---------------------

Example:

.. code-block:: c++
   :linenos:

    void reshape_(const vector<int> &new_shape);
    static Tensor* reshape(Tensor *A, const vector<int> &shape);
    static Tensor* flatten(Tensor *A);


Transpose-like operations
--------------------------

Example:

.. code-block:: c++
   :linenos:

    static Tensor* permute(Tensor* t, const vector<int>& dims);
    static Tensor* moveaxis(Tensor* t, int source, int destination);
    static Tensor* swapaxis(Tensor* t, int axis1, int axis2);


Changing number of dimensions
-------------------------------

Example:

.. code-block:: c++
   :linenos:

    static Tensor* squeeze(Tensor *A);
    static Tensor* unsqueeze(Tensor *A);


Joining arrays
---------------

Example:

.. code-block:: c++
   :linenos:

    static Tensor* concat(const vector<Tensor*> t, unsigned int axis=0, Tensor* output=nullptr);
    

Rearranging elements and transformations
-----------------------------------------

Example:

.. code-block:: c++
   :linenos:

    static void shift(Tensor *A,Tensor *B, vector<int> shift, string mode="constant", float constant=0.0f);
    static void rotate(Tensor *A, Tensor *B, float angle, vector<int> offset_center={0,0}, string mode="constant", float constant=0.0f);
    static void scale(Tensor *A, Tensor *B, vector<int> new_shape, string mode="nearest", float constant=0.0f);
    static void flip(Tensor *A, Tensor *B, int axis=0);
    static void crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant=0.0f);
    static void crop_scale(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, string mode="nearest", float constant=0.0f);
    static void cutout(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant=0.0f);
    static void shift_random(Tensor *A,Tensor *B, vector<float> factor_x, vector<float> factor_y, string mode="constant", float constant=0.0f);
    static void rotate_random(Tensor *A, Tensor *B, vector<float> factor, vector<int> offset_center={0,0}, string mode="constant", float constant=0.0f);
    static void scale_random(Tensor *A, Tensor *B, vector<float> factor, string mode="nearest", float constant=0.0f);
    static void flip_random(Tensor *A, Tensor *B, int axis);
    static void crop_random(Tensor *A, Tensor *B);
    static void crop_scale_random(Tensor *A, Tensor *B, vector<float> factor, string mode="nearest", float constant=0.0f);
    static void cutout_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant=0.0f);

   