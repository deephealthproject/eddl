Activations
============

Softmax
--------

.. doxygenfunction:: eddl::Softmax

Example:

.. code-block:: c++
   :linenos:

   layer Softmax(layer parent, string name="");


Sigmoid
--------

.. doxygenfunction:: eddl::Sigmoid

Example:

.. code-block:: c++
   :linenos:

   layer Sigmoid(layer parent, string name="");


ReLu
--------

.. doxygenfunction:: eddl::ReLu


Example:

.. code-block:: c++
   :linenos:

   layer ReLu(layer parent, string name="");



Threshold ReLu
---------------


Applies the Thresholded version of a Rectified Linear Unit activation function to the given layer.

Example:

.. code-block:: c++
   :linenos:

    layer ThresholdedReLu(layer parent, float alpha=1.0, string name="");




Leaky ReLu
-----------


Applies the Leaky version of a Rectified Linear Unit activation function to the given layer.

Example:

.. code-block:: c++
   :linenos:

    layer LeakyReLu(layer parent, float alpha=0.01, string name="");




ELu
--------

Applies the Exponential Linear Unit activation function to the given layer.

Example:

.. code-block:: c++
   :linenos:

    layer Elu(layer parent, float alpha=1.0, string name="");

SeLu
--------

Applies the Scaled Exponential Linear Unit activation function to the given layer.

Example:

.. code-block:: c++
   :linenos:

    layer Selu(layer parent, string name="");


Exponential
------------

Applies the Exponential (base e) activation function to the given layer.

Example:

.. code-block:: c++
   :linenos:

    layer Exponential(layer parent, string name="");




Softplus
------------

Applies the Softplus activation function to the given layer.

Example:

.. code-block:: c++
   :linenos:

    layer Softplus(layer parent, string name="");





Softsign
------------

Applies the Softsign activation function to the given layer.

Example:

.. code-block:: c++
   :linenos:

    layer Softsign(layer parent, string name="");





Linear
-------

Applies the Linear activation function to the given layer.

Example:

.. code-block:: c++
   :linenos:

    layer Linear(layer parent, float alpha=1.0, string name="");



Tanh
------


.. doxygenfunction:: eddl::Tanh

Example:

.. code-block:: c++
   :linenos:

    layer Tanh(layer parent, string name="");
