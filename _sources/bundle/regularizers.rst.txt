Regularizers
=============

L1
---

.. doxygenfunction:: L1


Example:

.. code-block:: c++

   l = L1(l, 0.0001);


L2
---

.. doxygenfunction:: L2

Example:

.. code-block:: c++

   l = L2(l, 0.0001);


L1L2
-----

.. doxygenfunction:: L1L2

Example:

.. code-block:: c++

   l = L1(l, 0.00001, 0.0001);

