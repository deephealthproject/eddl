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

    layer GlobalMaxPool(layer parent, string name = ""); //Todo: Implement



AveragePooling
--------------

.. doxygenfunction:: AveragePool

.. note::

    Not yet implemented.

Example:

.. code-block:: c++
   :linenos:

    layer AveragePool(layer parent, const vector<int> &pool_size = {2, 2}, const vector<int> &strides = {2, 2},string padding = "none", string name = "");



GlobalAveragePooling
--------------------

.. doxygenfunction:: GlobalAveragePool


.. note::

    Not yet implemented.

Example:

.. code-block:: c++
   :linenos:

    layer GlobalAveragePool(layer parent, string name = ""); //Todo: Implement


