Losses
=============

.. doxygenfunction:: eddl::getLoss

Mean Squared Error
------------------


Aliases: ``mse`` and ``categorical_accuracy``.

Example:

.. code-block:: c++
   :linenos:

    Loss* getLoss("mse");


Cross-Entropy
--------------------

Aliases: ``cross_entropy``.

Example:

.. code-block:: c++
   :linenos:

    Loss* getLoss("cross_entropy");


Soft Cross-Entropy
-------------------

Aliases: ``soft_cross_entropy``.

Example:

.. code-block:: c++
   :linenos:

    Loss* getLoss("soft_cross_entropy");

