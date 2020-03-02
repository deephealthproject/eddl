Test & Score
============

Evaluate model
--------------

.. doxygenfunction:: eddl::evaluate


Example:

.. code-block:: c++
   :linenos:

    void evaluate(model m, const vector<Tensor *> &in, const vector<Tensor *> &out);
    //e.g.: evaluate(mymodel, {X_test}, {Y_test});
