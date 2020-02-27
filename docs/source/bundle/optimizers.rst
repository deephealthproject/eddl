Optimizers
============

Adadelta
--------

Adadelta optimizer.

Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving window of gradient updates, instead of accumulating all past gradients. This way, Adadelta continues learning even when many updates have been done.

  Parameters:

    - ``lr``: A float. Initial learning rate.
    - ``rho``: A float. Adadelta decay factor, corresponding to fraction of gradient to keep at each time step.
    - ``epsilon``: A float. Term added to the denominator to improve numerical stability 
    - ``weight_decay``: A float. Weight decay (L2 penalty) 


Example:

.. code-block:: c++
   :linenos:

    optimizer adadelta(float lr, float rho, float epsilon, float weight_decay);


Adam
-----

Adam optimizer.

  Parameters:

    - ``lr``: A float. Initial learning rate.
    - ``beta_1, beta_2``: Floats. coefficients used for computing running averages of gradient and its square.
    - ``epsilon``: A float. Term added to the denominator to improve numerical stability 
    - ``weight_decay``: A float. Weight decay (L2 penalty).
    - ``amsgrad``: Boolean. Whether to use the AMSGrad variant of this algorithm.


Example:

.. code-block:: c++
   :linenos:

    optimizer adam(float lr=0.01, float beta_1=0.9, float beta_2=0.999, float epsilon=0.000001, float weight_decay=0,bool amsgrad=false);


Adagrad
----------

Adagrad optimizer.

  Parameters:

    - ``lr``: A float. Initial learning rate.
    - ``epsilon``: A float. Term added to the denominator to improve numerical stability 
    - ``weight_decay``: A float. Weight decay (L2 penalty).

Example:

.. code-block:: c++
   :linenos:

    optimizer adagrad(float lr, float epsilon, float weight_decay);

Adamax
----------

Adamax optimizer.

  Parameters:

    - ``lr``: A float. Initial learning rate.
    - ``beta_1, beta_2``: Floats. coefficients used for computing running averages of gradient and its square.
    - ``epsilon``: A float. Term added to the denominator to improve numerical stability 
    - ``weight_decay``: A float. Weight decay (L2 penalty).

Example:

.. code-block:: c++
   :linenos:

    optimizer adamax(float lr, float beta_1, float beta_2, float epsilon, float weight_decay);


Nadam
----------

Nesterov Adam optimizer.

Much like Adam is essentially RMSprop with momentum, Nadam is RMSprop with Nesterov momentum.

  Parameters:

    - ``lr``: A float. Initial learning rate.
    - ``beta_1, beta_2``: Floats. coefficients used for computing running averages of gradient and its square.
    - ``epsilon``: A float. Term added to the denominator to improve numerical stability 
    - ``schedule_decay``: A float.

Example:

.. code-block:: c++
   :linenos:

    optimizer nadam(float lr, float beta_1, float beta_2, float epsilon, float schedule_decay);


RMSProp
----------

RMSProp optimizer.

  Parameters:

    - ``lr``: A float. Initial learning rate.
    - ``rho``: A float. Decay factor, corresponding to fraction of gradient to keep at each time step.
    - ``epsilon``: A float. Term added to the denominator to improve numerical stability 
    - ``weight_decay``: A float. Weight decay (L2 penalty).


Example:

.. code-block:: c++
   :linenos:

    optimizer rmsprop(float lr=0.01, float rho=0.9, float epsilon=0.00001, float weight_decay=0.0);


SGD (Stochastic Gradient Descent)
----------------------------------

SGD optimizer.

Includes support for momentum, learning rate decay, and Nesterov momentum.

  Parameters:

    - ``lr``: A float. Initial learning rate.
    - ``momentum``: A float. Parameter that accelerates SGD in the relevant direction and dampens oscillations.
    - ``weight_decay``: A float. Weight decay (L2 penalty).
    - ``nesterov``: Boolean. Whether to apply Nesterov momentum.


Example:

.. code-block:: c++
   :linenos:

    optimizer sgd(float lr = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f, bool nesterov = false);

