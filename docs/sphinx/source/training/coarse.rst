Coarse training
===============

Fit
---

.. doxygenfunction:: eddl::fit


Example:

.. code-block:: c++
    
    model net = Model({in}, {out});

    // Build model
    ...

    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);

