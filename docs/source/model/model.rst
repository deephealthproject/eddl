Model
=====


Constructor
------------

Instantiates model, taking two vectors, one of input layers and another of output layers.

Example:

.. code-block:: c++
   :linenos:

    model Model(vlayer in, vlayer out);


Build
----------

Tell the model which optimizer, losses, metrics and computing services use.

.. note::

    Parameters:

    - ``net``: Model
    - ``o`` : Optimizer
    - ``lo`` : Vector with losses
    - ``me`` : Vector with metrics
    - ``cs`` : Computing service

Example:

.. code-block:: c++
   :linenos:

    void build(model net, optimizer o, const vector<string> &lo, const vector<string> &me, CompServ *cs=nullptr, bool init_weights=true);
    //e.g.: build(mymodel, sgd(0.01f), {"cross_entropy"}, {"accuracy"}, CS_GPU({1, 0}), true);


Summary
----------

Prints a summary representation of your model.

Example:

.. code-block:: c++
   :linenos:

    void summary(model m);


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

Plots a representation of your model.

Example:

.. code-block:: c++
   :linenos:

    void plot(model m, string fname, string mode="LR");

Result:

.. image:: /_static/images/models/mlp.svg



Load
--------------

Load weights to reinstantiate your model.

Example:

.. code-block:: c++
   :linenos:

    void load(model m, const string& fname, string format="bin");



Save
--------------------

Save weights of a model.

Example:

.. code-block:: c++
   :linenos:

    void save(model m, const string& fname, string format="bin");



Learning rate (on the fly)
--------------------------

Changes the learning rate and hyperparameters of the model optimizer.

Example:

.. code-block:: c++
   :linenos:

    void setlr(model net,vector<float>p);




Logging
--------

Save the training outputs of a model to a filename

Example:

.. code-block:: c++
   :linenos:

    void setlogfile(model net, string fname);




Move to device
---------------

Move the model to a specific device

Example:

.. code-block:: c++
   :linenos:

    void toCPU(model net, int t=std::thread::hardware_concurrency()); // num. threads, memory consumption
    void toGPU(model net, vector<int> g, int lsb, string mem); // mode, list of gpus (on=1/off=0), sync number, memory consumption

