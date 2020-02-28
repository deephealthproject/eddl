Activations
============

Softmax
--------

.. doxygenfunction:: eddl::Softmax

Example:

.. code-block:: c++
   

   layer Softmax(layer parent, string name="");


Sigmoid
--------

.. doxygenfunction:: eddl::Sigmoid

Example:

.. code-block:: c++
   

   layer Sigmoid(layer parent, string name="");


ReLu
--------

.. doxygenfunction:: eddl::ReLu


Example:

.. code-block:: c++
   

   layer ReLu(layer parent, string name="");



Threshold ReLu
---------------


.. doxygenfunction:: eddl::ThresholdedReLu

Example:

.. code-block:: c++
   

    layer ThresholdedReLu(layer parent, float alpha=1.0, string name="");




Leaky ReLu
-----------


.. doxygenfunction:: eddl::LeakyReLu

Example:

.. code-block:: c++
   

    layer LeakyReLu(layer parent, float alpha=0.01, string name="");




ELu
--------

.. doxygenfunction:: eddl::Elu

Example:

.. code-block:: c++
   

    layer Elu(layer parent, float alpha=1.0, string name="");

SeLu
--------

.. doxygenfunction:: eddl::Selu

Example:

.. code-block:: c++
   

    layer Selu(layer parent, string name="");


Exponential
------------

.. doxygenfunction:: eddl::Exponential

Example:

.. code-block:: c++
   

    layer Exponential(layer parent, string name="");




Softplus
------------

.. doxygenfunction:: eddl::Softplus

Example:

.. code-block:: c++
   

    layer Softplus(layer parent, string name="");





Softsign
------------

.. doxygenfunction:: eddl::Softsign

Example:

.. code-block:: c++
   

    layer Softsign(layer parent, string name="");





Linear
-------

.. doxygenfunction:: eddl::Linear

Example:

.. code-block:: c++
   

    layer Linear(layer parent, float alpha=1.0, string name="");



Tanh
------


.. doxygenfunction:: eddl::Tanh

Example:

.. code-block:: c++
   

    layer Tanh(layer parent, string name="");
