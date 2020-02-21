Pooling
=============

MaxPooling
----------

Max pooling operation

Example:

.. code-block:: c++
   :linenos:

    layer MaxPool(layer parent, const vector<int> &pool_size = {2, 2}, const vector<int> &strides = {2, 2}, string padding = "none", string name = "");



GlobalMaxPooling
-----------------

Global max pooling operation

Example:

.. code-block:: c++
   :linenos:

    layer GlobalMaxPool(layer parent, string name = ""); //Todo: Implement



AveragePooling
--------------

Average pooling operation

.. note::

    Not yet implemented.

Example:

.. code-block:: c++
   :linenos:

    layer AveragePool(layer parent, const vector<int> &pool_size = {2, 2}, const vector<int> &strides = {2, 2},string padding = "none", string name = "");



GlobalAveragePooling
--------------------

Global average pooling operation


.. note::

    Not yet implemented.

Example:

.. code-block:: c++
   :linenos:

    layer GlobalAveragePool(layer parent, string name = ""); //Todo: Implement


