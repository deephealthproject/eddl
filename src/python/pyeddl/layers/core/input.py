from pyeddl.layers.base import Layer


class Input(Layer):
    """Layer to be used as an entry point into a model.

    It can either wrap an existing tensor (pass an `input_tensor` argument)
    or create its a placeholder tensor (pass arguments `input_shape`
    or `batch_input_shape` as well as `dtype`).

    Args:
        input_shape: Shape tuple, not including the batch axis.
        batch_size: Optional input batch size (integer or None).
        batch_input_shape: Shape tuple, including the batch axis.
        dtype: Datatype of the input.
        input_tensor: Optional tensor to use as layer input
            instead of creating a placeholder.
        sparse: Boolean, whether the placeholder created
            is meant to be sparse.
        name: Name of the layer (string).
    """

    def __init__(self, input_shape=None, batch_size=None,
                 batch_input_shape=None,
                 dtype=None, input_tensor=None, sparse=False, name=None):
        super(Input, self).__init__()
