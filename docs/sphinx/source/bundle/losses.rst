Losses
=============

.. doxygenfunction:: eddl::getLoss


Mean Squared Error
------------------

Aliases: ``mean_squared_error`` and ``mse``.
Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input

Example:

.. code-block:: c++


   myloss = getLoss("mean_squared_error");


Binary Cross-Entropy
--------------------------

Aliases: ``binary_cross_entropy``, and ``bce``.
Creates a criterion that measures the Binary Cross Entropy between the target and the output. Values are encoded as
the probability of the positive class.

Example:

.. code-block:: c++

   myloss = getLoss("binary_cross_entropy");


Categorical Cross-Entropy
--------------------------

Aliases: ``categorical_cross_entropy``, ``cce``, ``cross_entropy``, and ``ce``.
Creates a criterion that measures the Categorical Cross Entropy between the target and the output.  Values are encoded as
vector of probabilities that sum is equal to one.

Example:

.. code-block:: c++

   myloss = getLoss("categorical_cross_entropy");


Softmax Cross-Entropy
-------------------

Aliases: ``softmax_cross_entropy``, ``soft_cross_entropy``, and ``sce``.

This is the optimized version of the CategoricalCrossEntropy when the last layer is a Softmax layer. It bypasses the
backward of the Softmax layer by applying the simplified derivative of ``f(x)=CE(Softmax(x)); df/dx= y_pred-y_target``.

Example:

.. code-block:: c++
   
   myloss = getLoss("softmax_cross_entropy");


Dice
-------------------

Alias: ``dice``.

Example:

.. code-block:: c++
   
   myloss = getLoss("dice");
