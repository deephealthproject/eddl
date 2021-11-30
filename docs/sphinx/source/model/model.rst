Model
=====


Constructor
------------


.. doxygenfunction:: eddl::Model(vlayer in, vlayer out)

Example:

.. code-block:: c++

    layer in1 = Input({3,32,32});
    layer in2 = Input({1,32,32});
    layer l = Concat(in1, in2);
    ...
    layer out = Activation(Dense(l, num_classes), "softmax");
    ...
    model net = Model({in1, in2}, {out});




Build
----------

.. doxygenfunction:: eddl::build(model net, optimizer o, const vector<string> &lo, const vector<string> &me, CompServ *cs = nullptr, bool init_weights = true)



Example:

.. code-block:: c++

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

    ...
    model net=Model({in},{out});

    plot(net,"model.pdf");

Result:

.. image:: /_static/images/models/mlp.svg



Load weights
--------------

Loads the weights of a model (not the topology).

.. doxygenfunction:: eddl::load(model m, const string& fname, const string& format="bin");

Example:

.. code-block:: c++

    load(net, "model-weights.bin");


Save weights
-------------

Save the weights of a model (not the topology).

.. doxygenfunction:: eddl::save(model m, const string& fname, const string& format="bin");

Example:

.. code-block:: c++

    save(net, "model-weights.bin");


Learning rate (on the fly)
--------------------------


.. doxygenfunction:: eddl::setlr(model, vector<float>)

Example:

.. code-block:: c++

    ...
    model net = Model({in}, {out});

    // Build model
    ...

    setlr(net, {0.005, 0.9});

    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);




Logging
--------


.. doxygenfunction:: eddl::setlogfile(model net, const string& fname);

Example:

.. code-block:: c++

    model net = Model({in}, {out});

    // Build model
    ...

    setlogfile(net,"model-log");

    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);




Move to device
---------------

Move the model to a specific device

.. doxygenfunction:: toCPU(model net, int th=-1)

Example:

.. code-block:: c++

    toCPU(net);


.. doxygenfunction:: toGPU(model net, vector<int> g={1}, int lsb=1, const string& mem="full_mem")

Example:

.. code-block:: c++
    
    toGPU(net, {1}); // Use GPU #1 (implicit: syncronize every batch and use 'full_mem' setup)
    toGPU(net, {1, 1}, 100, "low_mem"); // Use GPU #1 and #2, syncronize every 100 batches and use 'low_mem' setup


Get parameters
---------------

.. doxygenfunction:: eddl::get_parameters

Example:

.. code-block:: c++

    vector<vtensor> get_parameters(net, true); // deep_copy=true


Set parameters
---------------

.. doxygenfunction:: eddl::set_parameters

Example:

.. code-block:: c++

    vector<vtensor> myparameters = ...
    set_parameters(net, myparameters);