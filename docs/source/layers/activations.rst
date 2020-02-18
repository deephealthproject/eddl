Activations
============

Softmax
--------

Applies a Softmax activation function to the given layer.

Example:

.. code-block:: c++
   :linenos:

   layer Softmax(layer parent, string name="");


Sigmoid
--------

Applies a Sigmoid activation function to the given layer.

Example:

.. code-block:: c++
   :linenos:

   layer Sigmoid(layer parent, string name="");


ReLu
--------

Rectified Linear Unit activation function to the given layer.

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

Applies a Exponential Linear Unit activation function to the given layer.

Example:

.. code-block:: c++
   :linenos:

    layer Elu(layer parent, float alpha=1.0, string name="");

SeLu
--------

Applies the Scaled version of a Exponential Linear Unit activation function to the given layer.

Example:

.. code-block:: c++
   :linenos:

    layer Selu(layer parent, string name="");


Exponential
------------

Exponential (base e) activation function.

Example:

.. code-block:: c++
   :linenos:

    layer Exponential(layer parent, string name="");




Softplus
------------

Softplus activation function.

Example:

.. code-block:: c++
   :linenos:

    layer Softplus(layer parent, string name="");





Softplus
------------

Softsign activation function.

Example:

.. code-block:: c++
   :linenos:

    layer Softsign(layer parent, string name="");





Linear
-------

Linear activation function.

Example:

.. code-block:: c++
   :linenos:

    layer Linear(layer parent, float alpha=1.0, string name="");



Tanh
------

Hyperbolic tangent activation function.

Example:

.. code-block:: c++
   :linenos:

    layer Tanh(layer parent, string name="");
