Optimizers
============

Adadelta
--------

Adadelta optimizer.

Example:

.. code-block:: c++
   :linenos:

    optimizer adadelta(float lr, float rho, float epsilon, float weight_decay);


Adam
-----

Adadelta optimizer.

Example:

.. code-block:: c++
   :linenos:

    optimizer adam(float lr=0.01, float beta_1=0.9, float beta_2=0.999, float epsilon=0.000001, float weight_decay=0,bool amsgrad=false);


Adagrad
----------

Adagrad optimizer.

Example:

.. code-block:: c++
   :linenos:

    optimizer adagrad(float lr, float epsilon, float weight_decay);

Adamax
----------

Adamax optimizer.

Example:

.. code-block:: c++
   :linenos:

    optimizer adamax(float lr, float beta_1, float beta_2, float epsilon, float weight_decay);


Nadam
----------

Nadam optimizer.

Example:

.. code-block:: c++
   :linenos:

    optimizer nadam(float lr, float beta_1, float beta_2, float epsilon, float schedule_decay);


RMSProp
----------

RMSProp optimizer.

Example:

.. code-block:: c++
   :linenos:

    optimizer rmsprop(float lr=0.01, float rho=0.9, float epsilon=0.00001, float weight_decay=0.0);


SGD (Stochastic Gradient Descent)
----------------------------------

SGD optimizer.

Example:

.. code-block:: c++
   :linenos:

    optimizer sgd(float lr = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f, bool nesterov = false);

