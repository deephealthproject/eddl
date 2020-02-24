Indexing routines
=================

.. note::

    Section in progress

    Read this: <https://github.com/deephealthproject/eddl/blob/master/eddl_progress_tensor.md>


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

Returns an array with the selected indices.
The indices must be specified as a vector of strings ({"0", ":5", ":", "3:6"}).

.. code-block:: c++
    :linenos:

    Tensor* select(const vector<string>& indices);


set_select
^^^^^^^^^^

Sets the elements in the array using the selected indices.
The indices must be specified as a vector of strings ({"0", ":5", ":", "3:6"}).

.. code-block:: c++
    :linenos:

    void set_select(const vector<string>& indices, Tensor *A);
    
