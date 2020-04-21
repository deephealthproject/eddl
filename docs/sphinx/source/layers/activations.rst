Activations
============

Softmax
--------

.. doxygenfunction:: eddl::Softmax

Example:

.. code-block:: c++
   
   ...
   l = Softmax(l);


Sigmoid
--------

.. doxygenfunction:: eddl::Sigmoid

Example:

.. code-block:: c++
   
   ...
   l = Sigmoid(l);


ReLu
--------

.. doxygenfunction:: eddl::ReLu


Example:

.. code-block:: c++
   
   ...
   l = ReLu(l);



Threshold ReLu
---------------


.. doxygenfunction:: eddl::ThresholdedReLu

Example:

.. code-block:: c++
   
   ...
    l = ThresholdedReLu(l, 1.0);




Leaky ReLu
-----------


.. doxygenfunction:: eddl::LeakyReLu

Example:

.. code-block:: c++
   
   ...
    l = LeakyReLu(l, 0.01);




ELu
--------

.. doxygenfunction:: eddl::Elu

Example:

.. code-block:: c++
   
   ...
    l = Elu(l, 1.0);

SeLu
--------

.. doxygenfunction:: eddl::Selu

Example:

.. code-block:: c++
   
   ...
    l = Selu(l);


Exponential
------------

.. doxygenfunction:: eddl::Exponential

Example:

.. code-block:: c++
   
   ...
    l = Exponential(l);




Softplus
------------

.. doxygenfunction:: eddl::Softplus

Example:

.. code-block:: c++
   
   ...
    l = Softplus(l);





Softsign
------------

.. doxygenfunction:: eddl::Softsign

Example:

.. code-block:: c++
   
   ...
    l = Softsign(l);





Linear
-------

.. doxygenfunction:: eddl::Linear

Example:

.. code-block:: c++
   
   ...
    l = Linear(l, 1.0);



Tanh
------


.. doxygenfunction:: eddl::Tanh

Example:

.. code-block:: c++
   
   ...
    l = Tanh(l);
