from pyeddl.layers.base import Layer


class Activation(Layer):
    """Applies an activation function to an output.

    Args:
        activation: name of activation function to use

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.

    """

    def __init__(self, activation, **kwargs):
        super(Activation, self).__init__()
        self.supports_masking = True
        self.activation = None