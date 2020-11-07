Losses
=============

.. doxygenfunction:: eddl::getLoss


Mean Squared Error
------------------

Aliases: ``mse`` and ``mean_squared_error``.

Example:

.. code-block:: c++

   my_loss = getLoss("mse");
   // or
   my_loss = getLoss("mean_squared_error");


Categorical Cross-Entropy
--------------------------

Alias: ``categorical_cross_entropy``.

Example:

.. code-block:: c++

   my_loss = getLoss("categorical_cross_entropy");


Binary Cross-Entropy
--------------------------

Alias: ``binary_cross_entropy``.

Example:

.. code-block:: c++

   my_loss = getLoss("binary_cross_entropy");


Cross-Entropy
--------------------

Alias: ``cross_entropy``.

Example:

.. code-block:: c++

   my_loss = getLoss("cross_entropy");


Soft Cross-Entropy
-------------------

Alias: ``soft_cross_entropy``.

Example:

.. code-block:: c++
   
   my_loss = getLoss("soft_cross_entropy");


Dice
-------------------

Alias: ``dice``.

Example:

.. code-block:: c++
   
   my_loss = getLoss("dice");
