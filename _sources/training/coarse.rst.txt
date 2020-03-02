Coarse training
===============

Fit
---

.. doxygenfunction:: eddl::fit


Example:

.. code-block:: c++

    void fit(model m, const vector<Tensor *> &in, const vector<Tensor *> &out, int batch, int epochs);
    // e.g.: fit(mymodel, {X_train}, {Y_train}, 128, 25);

