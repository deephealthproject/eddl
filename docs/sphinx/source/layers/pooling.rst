Pooling
=============

MaxPooling1D
--------------

.. doxygenfunction:: MaxPool1D

Example:

.. code-block:: c++

    l = MaxPool1D(l, {4}, {4}, "same");


MaxPooling2D
-------------

.. doxygenfunction:: MaxPool2D


Example:

.. code-block:: c++

    l = MaxPool2D(l, {3, 3}, {1, 1}, "same");
    

MaxPooling3D
--------------

.. doxygenfunction:: MaxPool3D


.. code-block:: c++

    l = MaxPool3D(l, {3, 3, 3}, {3, 3, 3}, "same");



AveragePooling1D
------------------

.. doxygenfunction:: AveragePool1D

Example:

.. code-block:: c++

    l = AveragePool1D(l, {4}, {4}, "same");


AveragePooling2D
------------------

.. doxygenfunction:: AveragePool2D


Example:

.. code-block:: c++

    l = AveragePool2D(l, {3, 3}, {1, 1}, "same");


AveragePooling3D
------------------

.. doxygenfunction:: AveragePool3D


.. code-block:: c++

    l = AveragePool3D(l, {3, 3, 3}, {3, 3, 3}, "same");



GlobalMaxPooling1D
-------------------

.. doxygenfunction:: GlobalMaxPool1D

Example:

.. code-block:: c++

    l = GlobalMaxPool1D(l);


GlobalMaxPooling2D
-------------------

.. doxygenfunction:: GlobalMaxPool2D


Example:

.. code-block:: c++

    l = GlobalMaxPool2D(l);


GlobalMaxPooling3D
-------------------

.. doxygenfunction:: GlobalMaxPool3D


.. code-block:: c++

    l = GlobalMaxPool3D(l);



GlobalAveragePooling1D
-----------------------

.. doxygenfunction:: GlobalAveragePool1D

Example:

.. code-block:: c++

    l = GlobalAveragePool1D(l);


GlobalAveragePooling2D
-----------------------

.. doxygenfunction:: GlobalAveragePool2D


Example:

.. code-block:: c++

    l = GlobalAveragePool2D(l);


GlobalAveragePooling3D
-----------------------

.. doxygenfunction:: GlobalAveragePool3D


.. code-block:: c++

    l = GlobalAveragePool3D(l);



