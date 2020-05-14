Pooling
=============

MaxPooling1D
------------

.. doxygenfunction:: MaxPool1D

Example:

.. code-block:: c++
    :linenos:

    ...
    l = Reshape(l,{1,784}); //image as a 1D signal with depth=1
    l = Conv1D(l,16, {3},{1});
    l = ReLu(l);
    l = MaxPool1D(l,{4},{4});  //MaxPool 4 stride 4
    ...


MaxPooling
----------

.. doxygenfunction:: MaxPool


Example:

.. code-block:: c++
   :linenos:

    ...
    l = Reshape(l,{1,28,28});
    l = Conv(l,32, {3,3},{1,1});
    l = ReLu(l);
    l = MaxPool(l,{3,3}, {1,1}, "same");
    ...


GlobalMaxPooling
-----------------

.. doxygenfunction:: GlobalMaxPool


Example:

.. code-block:: c++
   :linenos:

    ...
    l = MaxPool(ReLu(Conv(l,32,{3,3},{1,1})),{2,2});
    l = MaxPool(ReLu(Conv(l,64,{3,3},{1,1})),{2,2});
    l = MaxPool(ReLu(Conv(l,128,{3,3},{1,1})),{2,2});
    l = GlobalMaxPool(l);
    ...


AveragePooling
--------------

.. doxygenfunction:: AveragePool

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#core-layers




GlobalAveragePooling
--------------------

.. doxygenfunction:: GlobalAveragePool


.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#core-layers

