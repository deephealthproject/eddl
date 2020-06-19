Model
=====


Constructor
------------


.. doxygenfunction:: eddl::Model(vlayer, vlayer)

Example:

.. code-block:: c++
   :linenos:

    layer in1 = Input({3,32,32});
    layer in2 = Input({1,32,32});
    layer l = Concat(in1, in2);
    ...
    layer out = Activation(Dense(l, num_classes), "softmax");
    ...
    model net = Model({in1, in2}, {out});


Build
----------


.. doxygenfunction:: eddl::build(model, optimizer, const vector<string>&, const vector<string>&, CompServ *, bool)


Example:

.. code-block:: c++
   :linenos:

    ...
    model net=Model({in},{out});

    // Build model
    build(net,
          sgd(0.01, 0.9), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}, "low_mem") // GPU with only one gpu
    );
    

Summary
----------

.. doxygenfunction:: eddl::summary

Example:

.. code-block:: c++
   :linenos:

    ...
    model net=Model({in},{out});

    // Build model
    build(net,
          sgd(0.01, 0.9), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}, "low_mem") // GPU with only one gpu
    );

    summary(net);


Result:

.. code-block:: text

    Generating Random Table
    ---------------------------------------------
    input1                        |  (784)     =>      (784)
    dense1                        |  (784)     =>      (1024)
    leaky_relu1                   |  (1024)    =>      (1024)
    dense2                        |  (1024)    =>      (1024)
    leaky_relu2                   |  (1024)    =>      (1024)
    dense3                        |  (1024)    =>      (1024)
    leaky_relu3                   |  (1024)    =>      (1024)
    dense4                        |  (1024)    =>      (10)
    softmax4                      |  (10)      =>      (10)
    ---------------------------------------------


Plot
-----------------


.. doxygenfunction:: eddl::plot

Example:

.. code-block:: c++
   :linenos:

    ...
    model net=Model({in},{out});

    plot(net,"model.pdf");

Result:

.. image:: /_static/images/models/mlp.svg



Load
--------------


.. doxygenfunction:: eddl::load(model, string&, string)

Example:

.. code-block:: c++
   :linenos:

    ...
    model net = Model({in}, {out});

    // Build model
    build(net,
          rmsprop(0.01), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
           CS_GPU({1,1},100) // one GPU
    );

    // Load weights
    load(net, "saved-weights.bin");

    // Evaluate
    evaluate(net, {x_test}, {y_test});


Save
--------------------


.. doxygenfunction:: eddl::save(model, string&, string)

Example:

.. code-block:: c++
   :linenos:

    ...
    model net = Model({in}, {out});

    // Build model
    build(net,
          rmsprop(0.01), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
           CS_GPU({1,1},100) // one GPU
    );
    
    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);

    // Save weights
    save(net, "saved-weights.bin");


Learning rate (on the fly)
--------------------------


.. doxygenfunction:: eddl::setlr(model, vector<float>)

Example:

.. code-block:: c++
   :linenos:

    ...
    model net = Model({in}, {out});

    // Build model
    ...

    setlr(net,{0.005,0.9});

    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);




Logging
--------


.. doxygenfunction:: eddl::setlogfile(model, string)

Example:

.. code-block:: c++
   :linenos:

    model net = Model({in}, {out});

    // Build model
    ...

    setlogfile(net,"model-log");

    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);




Move to device
---------------

Move the model to a specific device

.. doxygenfunction:: eddl::toCPU

Example:

.. code-block:: c++
   :linenos:

    toCPU(net);

.. doxygenfunction:: eddl::toGPU(model, vector<int>, int, string)

Example:

.. code-block:: c++
   :linenos:

    
    toGPU(net,{1},100,"low_mem"); // In two gpus, syncronize every 100 batches, low_mem setup
