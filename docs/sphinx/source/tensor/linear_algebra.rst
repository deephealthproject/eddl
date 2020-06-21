Linear algebra
==============

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/eddl_progress_tensor.md


Matrix and vector operations
-------------------------------

interpolate
^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::interpolate(float, Tensor *, float, Tensor *)
.. doxygenfunction:: Tensor::interpolate(float, Tensor *, float, Tensor *, Tensor *)

Example:

.. code-block:: c++
   :linenos:

    static Tensor* interpolate(float factor1, Tensor *A, float factor2, Tensor *B); // (new)C = f1*A + f2*B
    static void interpolate(float factor1, Tensor *A, float factor2, Tensor *B, Tensor *C);  // C = f1*A + f2*B


trace
^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::trace(int)
.. doxygenfunction:: Tensor::trace(Tensor *, int)

Example:

.. code-block:: c++
   :linenos:

    float trace(int k=0);
    static float trace(Tensor *A, int k=0);


norm
^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::norm(string)
.. doxygenfunction:: Tensor::norm(Tensor *, string)
.. doxygenfunction:: Tensor::norm(vector<int>, bool, string)

Example:

.. code-block:: c++
   :linenos:

    float norm(string ord="fro");
    static float norm(Tensor *A, string ord="fro");
    Tensor* norm(vector<int> axis, bool keepdims, string ord="fro");


