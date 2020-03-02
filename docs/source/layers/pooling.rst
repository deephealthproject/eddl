Pooling
=============

MaxPooling
----------

.. doxygenfunction:: MaxPool


Example:

.. code-block:: c++
   :linenos:

    layer MaxPool(layer parent, const vector<int> &pool_size = {2, 2}, const vector<int> &strides = {2, 2}, string padding = "none", string name = "");



GlobalMaxPooling
-----------------

.. doxygenfunction:: GlobalMaxPool


Example:

.. code-block:: c++
   :linenos:

    layer GlobalMaxPool(layer parent, string name = "");



AveragePooling
--------------

.. doxygenfunction:: AveragePool

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#core-layers

Example:

.. code-block:: c++
   :linenos:

    layer AveragePool(layer parent, const vector<int> &pool_size = {2, 2}, const vector<int> &strides = {2, 2},string padding = "none", string name = "");



GlobalAveragePooling
--------------------

.. doxygenfunction:: GlobalAveragePool


.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#core-layers

Example:

.. code-block:: c++
   :linenos:

    layer GlobalAveragePool(layer parent, string name = "");


