Test & Score
============

Evaluate model
--------------

Returns the loss value & metrics values for the model in test mode.

.. note::

    Parameters:

    - ``m``: Model
    - ``in`` : Input data (features)
    - ``out`` : Output data (labels)

Example:

.. code-block:: c++
   :linenos:

    void evaluate(model m, const vector<Tensor *> &in, const vector<Tensor *> &out);
    //e.g.: evaluate(mymodel, {X_test}, {Y_test});
