Indexing routines
=================

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/eddl_progress_tensor.md


Generating index arrays
-----------------------

...

Example:

.. code-block:: c++
    :linenos:

    ...


Indexing-like operations
-------------------------

select
^^^^^^

.. doxygenfunction:: Tensor::select(const vector<string>&)

.. code-block:: c++
    :linenos:

    Tensor* select(const vector<string>& indices);


set_select
^^^^^^^^^^

.. doxygenfunction:: Tensor::set_select(const vector<string>&, Tensor *)

.. code-block:: c++
    :linenos:

    void set_select(const vector<string>& indices, Tensor *A);
    
