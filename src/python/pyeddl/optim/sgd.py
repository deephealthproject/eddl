from pyeddl.optim import Optimizer


class SGD(Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    Args:
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.

    """

    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.nesterov = nesterov

