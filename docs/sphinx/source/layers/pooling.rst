Pooling
=============

MaxPooling1D
------------

.. doxygenfunction:: MaxPool1D

Example:

.. code-block:: c++

    l = MaxPool1D(l,{4},{4});  


MaxPooling
----------

.. doxygenfunction:: MaxPool


Example:

.. code-block:: c++

    l = MaxPool(l,{3,3}, {1,1}, "same");
    


GlobalMaxPooling
-----------------

.. doxygenfunction:: GlobalMaxPool


Example:

.. code-block:: c++

    l = GlobalMaxPool(l);
    


AveragePooling
--------------

.. doxygenfunction:: AveragePool



GlobalAveragePooling
--------------------

.. doxygenfunction:: GlobalAveragePool

.. code-block:: c++

    l = GlobalAveragePool(l);
    


