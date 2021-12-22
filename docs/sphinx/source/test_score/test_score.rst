Test & Score
============

Evaluate model
--------------

.. doxygenfunction:: eddl::evaluate


Example:

.. code-block:: c++

    evaluate(mymodel, {X_test}, {Y_test});

Make predictions
----------------

.. doxygenfunction:: eddl::predict

Example:

.. code-block:: c++

    preds = predict(mymodel, {X_test});
