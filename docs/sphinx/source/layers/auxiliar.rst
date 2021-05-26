Auxiliar Layers
================

Constant Of Tensor
-------------------

.. doxygenfunction:: ConstOfTensor

Example:

.. code-block:: c++

    t = Tensor::ones({16, 16, 16}};
    l = ConstOfTensor(t);
    

Where
------------------

.. doxygenfunction:: Where

Example:

.. code-block:: c++

    l = Where(parent1, parent2, condition);
