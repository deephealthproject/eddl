from pyeddl.layers.base import Layer


class Concatenate(Layer):
    """Layer that concatenates a list of inputs.

    It takes as input a list of tensors,
    all of the same shape except for the concatenation axis,
    and returns a single tensor, the concatenation of all inputs.

    Args:
        axis: Axis along which to concatenate.
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, axis=-1, **kwargs):
        super(Concatenate, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True
        self._reshape_required = False