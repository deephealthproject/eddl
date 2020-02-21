Coarse training
===============

Fit
---

Trains the model for a fixed number of epochs (iterations on a dataset).

.. note::

    Parameters:

    - ``m``: Model
    - ``in`` : Input data (features)
    - ``out`` : Output data (labels)
    - ``batch`` : Number of samples per gradient update
    - ``epochs`` : Number of epochs to train the model. An epoch is an iteration over the entire data provided

Example:

.. code-block:: c++
   :linenos:

    void fit(model m, const vector<Tensor *> &in, const vector<Tensor *> &out, int batch, int epochs);
    // e.g.: fit(mymodel, {X_train}, {Y_train}, 128, 25);

