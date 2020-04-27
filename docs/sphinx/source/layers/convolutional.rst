Convolutions
============

Conv2D
--------

.. doxygenfunction:: eddl::Conv

Example:

.. code-block:: c++
   :linenos:
    
    ...
    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = Reshape(l,{1,28,28});
    l = Conv(l,32, {3,3},{1,1});
    l = ReLu(l);
    l = MaxPool(l,{3,3}, {1,1}, "same");
    l = Conv(l,64, {3,3},{1,1});
    l = ReLu(l);
    l = MaxPool(l,{2,2}, {2,2}, "none");
    l = Reshape(l,{-1});

    layer out = Activation(Dense(l, num_classes), "softmax");
    model net = Model({in}, {out});
    ...
    

2D Upsampling 
--------------

.. doxygenfunction:: eddl::UpSampling

.. note::

    In future versions this function will call ``scale`` instead of ``repeat``

Example:

.. code-block:: c++
   :linenos:
    
    ...
    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = Reshape(l,{1,28,28});
    l = MaxPool(ReLu(Conv(l,128,{3,3},{2,2})),{2,2});
    l = UpSampling(l, {2, 2});
    ...


Convolutional Transpose
------------------------

.. doxygenfunction:: eddl::ConvT

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#convolutional-layers

