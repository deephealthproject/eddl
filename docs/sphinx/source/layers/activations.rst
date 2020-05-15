Activations
============

Softmax
--------

.. doxygenfunction:: eddl::Softmax

The Softmax activation function is: ``softmax(x) = exp(x) / reduce_sum(exp(x))``

Example:

.. code-block:: c++
   :linenos:
   
   ...
   l = Softmax(l);


Sigmoid
--------

.. doxygenfunction:: eddl::Sigmoid

The Sigmoid activation function is: ``sigmoid(x) = 1 / (1 + exp(-x))``

Example:

.. code-block:: c++
   :linenos:
   
   ...
   l = Sigmoid(l);


ReLu
--------

.. doxygenfunction:: eddl::ReLu

The ReLu activation function is:

- ``if x > 0: relu(x) = x``

- ``else: relu(x) = 0``

Example:

.. code-block:: c++
   :linenos:
   
   ...
   l = ReLu(l);



Thresholded ReLu
---------------


.. doxygenfunction:: eddl::ThresholdedReLu

The Thresholded ReLu activation function is:

- if ``x > alpha``: ``threshdolded_relu(x) = x``

- else: ``thresholded_relu(x) = 0``

Example:

.. code-block:: c++
   :linenos:
   
   ...
    l = ThresholdedReLu(l, 1.0);




Leaky ReLu
-----------

.. doxygenfunction:: eddl::LeakyReLu

The Leaky ReLu activation function is:

- if ``x > 0``: ``leaky_relu(x) = x``

- else: ``leaky_relu(x) = alpha * x``

Example:

.. code-block:: c++
   :linenos:
   
   ...
    l = LeakyReLu(l, 0.01);




ELu
--------

.. doxygenfunction:: eddl::Elu

The ELu activation function is:

- if ``x > 0``: ``elu(x) = x``

- else: ``elu(x) = alpha *  (exp(x) - 1)``

Example:

.. code-block:: c++
   :linenos:
   
   ...
    l = Elu(l, 1.0);

SeLu
--------

.. doxygenfunction:: eddl::Selu

The SeLu activation function is:

- if ``x > 0``: ``selu(x) = scale * x``

- else: ``selu(x) = scale * (alpha *  (exp(x) - 1))``

where ``alpha = 1.6732632423543772848170429916717`` and ``scale = 1.0507009873554804934193349852946``

Example:

.. code-block:: c++
   :linenos:
   
   ...
    l = Selu(l);


Exponential
------------

.. doxygenfunction:: eddl::Exponential

The Exponential activation function is: ``exp(x)``

Example:

.. code-block:: c++
   :linenos:
   
   ...
    l = Exponential(l);




Softplus
------------

.. doxygenfunction:: eddl::Softplus

The Softplus activation function is: ``softplus(x) = log(1 + exp(x))``

Example:

.. code-block:: c++
   :linenos:
   
   ...
    l = Softplus(l);





Softsign
------------

.. doxygenfunction:: eddl::Softsign

The Softsign activation function is: ``softsign(x) = x / (1 + abs(x))``

Example:

.. code-block:: c++
   :linenos:
   
   ...
    l = Softsign(l);





Linear
-------

.. doxygenfunction:: eddl::Linear

The Linear activation function is: ``linear(x) = alpha * x``

Example:

.. code-block:: c++
   :linenos:
   
   ...
    l = Linear(l, 1.0);



Tanh
------

.. doxygenfunction:: eddl::Tanh

The Tanh activation function is: ``tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x)))``

Example:

.. code-block:: c++
   :linenos:
   
   ...
    l = Tanh(l);
