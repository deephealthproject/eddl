from pyeddl.layers.base import Layer


class Add(Layer):
    """Layer that adds a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).

    Example:
        ```python
            import pyeddl
            input1 = pyeddl.layers.Input(shape=(16,))
            x1 = pyeddl.layers.Dense(8, activation='relu')(input1)
            input2 = pyeddl.layers.Input(shape=(32,))
            x2 = pyeddl.layers.Dense(8, activation='relu')(input2)
            # equivalent to added = pyeddl.layers.add([x1, x2])
            added = pyeddl.layers.Add()([x1, x2])
            out = pyeddl.layers.Dense(4)(added)
            model = pyeddl.models.Model(inputs=[input1, input2], outputs=out)
        ```

    """