Optimizers
============

Adadelta
--------

.. doxygenfunction:: adadelta

Example:

.. code-block:: c++

    opt = adadelta(0.001, 0.95, 0.000001, 0);


Adam
-----


.. doxygenfunction:: adam

Example:

.. code-block:: c++

    opt = adam(0.001);


Adagrad
----------

.. doxygenfunction:: adagrad


Example:

.. code-block:: c++

    opt = adagrad(0.001, 0.000001, 0);


Adamax
----------

.. doxygenfunction:: adamax


Example:

.. code-block:: c++

    opt = adamax(0.001, 0.9, 0.999, 0.000001, 0);


Nadam
----------

.. doxygenfunction:: nadam


Example:

.. code-block:: c++

    opt = nadam(0.001, 0.9, 0.999, 0.0000001, 0.004);


RMSProp
----------

.. doxygenfunction:: rmsprop


Example:

.. code-block:: c++

    opt = rmsprop(0.001);


SGD (Stochastic Gradient Descent)
----------------------------------

.. doxygenfunction:: sgd

Example:

.. code-block:: c++

    opt = sgd(0.001);

