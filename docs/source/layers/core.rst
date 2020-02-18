Core
========

Dense
--------

Regular densely-connected NN layer.

Example:

.. code-block:: c++
   :linenos:

   layer Dense(layer parent, int ndim, bool use_bias = true,  string name = "");

Embedding
-----------

Turns positive integers (indexes) into dense vectors of fixed size. 

Example:

.. code-block:: c++
   :linenos:

    layer Embedding(int input_dim, int output_dim, string name = "");


Reshape
--------

Reshapes an output to a certain shape.

Example:

.. code-block:: c++
   :linenos:

    layer Reshape(layer parent, const vector<int> &shape, string name = "");


Transpose
----------

Transposes a layer.

Example:

.. code-block:: c++
   :linenos:

    layer Transpose(layer parent, string name = "");


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