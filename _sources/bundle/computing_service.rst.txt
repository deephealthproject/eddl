A computing service is an object that provides hardware transparency so you can easily change the hardware on which
the code will be executed.

CPU
====

.. doxygenfunction:: CS_CPU

Example:

.. code-block:: c++

    build(net,
          rmsprop(0.01),            // Optimizer
          {"soft_cross_entropy"},   // Losses
          {"categorical_accuracy"}, // Metrics
          CS_CPU(4),                // CPU with 4 threads
          false
    );


GPU
====

.. doxygenfunction:: eddl::CS_GPU(const vector<int>, int, string)
.. doxygenfunction:: eddl::CS_GPU(const vector<int>, string)

Example:

.. code-block:: c++

    build(imported_net,
          rmsprop(0.01),            // Optimizer
          {"soft_cross_entropy"},   // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}),              // one GPU
          false
    );


FPGA
====
..
.. .. doxygenfunction:: eddl::CS_FGPA(const vector<int> &f, int lsb=1)

.. note::

    **Not implemented yet**

.. Example:
..
.. .. code-block:: c++
..    :linenos:
.. 
..     compserv CS_FGPA(const vector<int> &f, int lsb=1);



COMPSS
======

.. doxygenfunction:: CS_COMPSS

.. note::

    **Not implemented yet**


