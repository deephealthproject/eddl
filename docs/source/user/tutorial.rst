Getting started
***************************


Multi-layer Perceptron
=======================


.. code-block:: python
   :linenos:

    from pyeddl.layers import Tensor, Input, Dense, Activation, Drop
    from pyeddl.models import Model
    from pyeddl.datasets import mnist


    # Get dataset
    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    # Input
    batch = 1000
    in_layer = Input(shape=(batch, 784))

    # Layers
    l1 = Activation(Dense(in_layer, 1024), 'relu')
    l2 = Activation(Dense(l1, 1024), 'relu')
    out_layer = Activation(Dense(Drop(l2, 0.5), 10), 'softmax')

    m = Model(in_layer, out_layer)

    # Plot model
    m.plot("model.pdf")

    # Get info
    m.summary()

    # Create optimizer, loss and metric
    opt = SGD(lr=0.01, mu=0.9)
    losses = [SoftCrossEntropy()]
    metrics = [CategoricalAccuracy()]

    # Define computing services (CPU, GPU, FPGA)
    cs = ComputingService(CPU_threads=4)

    # Build network
    m.build(opt, losses, metrics, cs)

    # Train model
    m.fit(x_train, y_train, batch=batch, epochs=1)

    # Evaluate model
    m.evaluate(x_train, y_train)



Convolutional
=======================


.. code-block:: python
   :linenos:

    from pyeddl.layers import Tensor, Input, Dense, Activation, Drop
    from pyeddl.models import Model
    from pyeddl.datasets import mnist


    # Get dataset
    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    # Input
    batch = 1000
    in_layer = Input(shape=(batch, 784))

    # Layers
    l = Reshape(in_layer, [batch, 1, 28, 28])
    l = MaxPool(Activation(Conv(l, 16, [3, 3]), 'relu'), [2, 2])
    l = MaxPool(Activation(Conv(l, 32, [3, 3]), 'relu'), [2, 2])
    l = MaxPool(Activation(Conv(l, 64, [3, 3]), 'relu'), [2, 2])
    l = MaxPool(Activation(Conv(l, 128, [3, 3]), 'relu'), [2, 2])
    l = Reshape(l, [batch, -1])
    l = Activation(Dense(l, 32), 'relu')
    out_tensor = Activation(Dense(l, 10), 'softmax')

    m = Model(in_layer, out_layer)

    # Plot model
    m.plot("model.pdf")

    # Get info
    m.summary()

    # Create optimizer, loss and metric
    opt = SGD(lr=0.01, mu=0.9)
    losses = [SoftCrossEntropy()]
    metrics = [CategoricalAccuracy()]

    # Define computing services (CPU, GPU, FPGA)
    cs = ComputingService(CPU_threads=4)

    # Build network
    m.build(opt, losses, metrics, cs)

    # Train model
    m.fit(x_train, y_train, batch=batch, epochs=1)

    # Evaluate model
    m.evaluate(x_train, y_train)