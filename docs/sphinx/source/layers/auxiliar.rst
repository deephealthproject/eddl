Auxiliar Layers
================

Constant Of Tensor
-------------------

.. doxygenfunction:: ConstOfTensor(Tensor* t, string name = "");

Example:

.. code-block:: c++

    // Example #1:
    Tensor *t = Tensor::ones({16, 16, 16}};
    Layer *l = ConstOfTensor(t);

    // Example #2:
    layer mean = ConstOfTensor(new Tensor( {0.485, 0.456, 0.406}, {3}, DEV_CPU));
    layer std = ConstOfTensor(new Tensor( {0.229, 0.224, 0.225}, {3}, DEV_CPU));

    

Where
------------------

.. doxygenfunction:: Where(layer parent1, layer parent2, layer condition, string name = "");

Example:

.. code-block:: c++

    l = Where(parent1, parent2, condition);
