Optimizers
============

Adadelta
--------

.. doxygenfunction:: adadelta

Example:

.. code-block:: c++

    optimizer adadelta(float lr, float rho, float epsilon, float weight_decay);


Adam
-----


.. doxygenfunction:: adam

Example:

.. code-block:: c++

    optimizer adam(float lr=0.01, float beta_1=0.9, float beta_2=0.999, float epsilon=0.000001, float weight_decay=0,bool amsgrad=false);


Adagrad
----------

.. doxygenfunction:: adagrad


Example:

.. code-block:: c++

    optimizer adagrad(float lr, float epsilon, float weight_decay);

Adamax
----------

.. doxygenfunction:: adamax


Example:

.. code-block:: c++

    optimizer adamax(float lr, float beta_1, float beta_2, float epsilon, float weight_decay);


Nadam
----------

.. doxygenfunction:: nadam


Example:

.. code-block:: c++

    optimizer nadam(float lr, float beta_1, float beta_2, float epsilon, float schedule_decay);


RMSProp
----------

.. doxygenfunction:: rmsprop


Example:

.. code-block:: c++

    optimizer rmsprop(float lr=0.01, float rho=0.9, float epsilon=0.00001, float weight_decay=0.0);


SGD (Stochastic Gradient Descent)
----------------------------------

.. doxygenfunction:: sgd

Example:

.. code-block:: c++

    optimizer sgd(float lr = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f, bool nesterov = false);

