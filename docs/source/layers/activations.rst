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

Applies the Thresholded version of a Rectified Linear Unit activation function to the given layer.

Example:

.. code-block:: c++
   :linenos:

   layer ReLu(layer parent, string name="");