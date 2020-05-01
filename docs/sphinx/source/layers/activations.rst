Activations
============

Softmax
--------

.. doxygenfunction:: eddl::Softmax

Example:

.. code-block:: c++
   :linenos:
   
   ...
   l = Softmax(l);


Sigmoid
--------

.. doxygenfunction:: eddl::Sigmoid

Example:

.. code-block:: c++
   :linenos:
   
   ...
   l = Sigmoid(l);


ReLu
--------

.. doxygenfunction:: eddl::ReLu


Example:

.. code-block:: c++
   :linenos:
   
   ...
   l = ReLu(l);



Threshold ReLu
---------------


.. doxygenfunction:: eddl::ThresholdedReLu

Example:

.. code-block:: c++
   :linenos:
   
   ...
    l = ThresholdedReLu(l, 1.0);




Leaky ReLu
-----------


.. doxygenfunction:: eddl::LeakyReLu

Example:

.. code-block:: c++
   :linenos:
   
   ...
    l = LeakyReLu(l, 0.01);




ELu
--------

.. doxygenfunction:: eddl::Elu

Example:

.. code-block:: c++
   :linenos:
   
   ...
    l = Elu(l, 1.0);

SeLu
--------

.. doxygenfunction:: eddl::Selu

Example:

.. code-block:: c++
   :linenos:
   
   ...
    l = Selu(l);


Exponential
------------

.. doxygenfunction:: eddl::Exponential

Example:

.. code-block:: c++
   :linenos:
   
   ...
    l = Exponential(l);




Softplus
------------

.. doxygenfunction:: eddl::Softplus

Example:

.. code-block:: c++
   :linenos:
   
   ...
    l = Softplus(l);





Softsign
------------

.. doxygenfunction:: eddl::Softsign

Example:

.. code-block:: c++
   :linenos:
   
   ...
    l = Softsign(l);





Linear
-------

.. doxygenfunction:: eddl::Linear

Example:

.. code-block:: c++
   :linenos:
   
   ...
    l = Linear(l, 1.0);



Tanh
------


.. doxygenfunction:: eddl::Tanh

Example:

.. code-block:: c++
   :linenos:
   
   ...
    l = Tanh(l);
