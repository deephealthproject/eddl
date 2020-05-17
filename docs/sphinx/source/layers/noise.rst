Noise Layers
=============

Gaussian Noise
---------------

.. doxygenfunction:: GaussianNoise

Example:

.. code-block:: c++
   :linenos:

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    // Data augmentation assumes 3D tensors... images:
    l=Reshape(l,{1,28,28});
    // Data augmentation
    l = RandomCropScale(l, {0.9f, 1.0f});

    // Come back to 1D tensor for fully connected:
    l=Reshape(l,{-1});
    l=Dense(l, 1024);
    l=BatchNormalization(l);
    l=GaussianNoise(l, 0.3);
    l=ReLu(l);
    ...

