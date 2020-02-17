Core
========

Dense
--------

Regular densely-connected NN layer.

Example:

.. code-block:: c++
   :linenos:

   layer Dense(layer parent, int ndim, bool use_bias = true,  string name = "");


Input
--------

Used to initialize an input to a model.

Example:

.. code-block:: c++
   :linenos:

   layer Input(const vector<int> &shape, string name = "");



Dropout
--------

Used to initialize an input to a model.

Example:

.. code-block:: c++
   :linenos:

   layer Dropout(layer parent, float rate, string name = "");