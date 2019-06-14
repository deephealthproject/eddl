import pyeddl
from pyeddl import _C


def _get_optim(optim):
    if isinstance(optim, pyeddl.optim.SGD):
        return _C.SGD(optim.lr, optim.momentum, optim.decay, optim.nesterov)
    else:
        NotImplementedError('optim')

def _get_loss(name):
    if name == 'mean_squared_error' or name == 'mse':
        return _C.LMeanSquaredErrorI()
    elif name == 'categorical_crossentropy' or name == 'crossentropy':
        return _C.LCrossEntropy()
    elif name == 'categorical_soft_crossentropy' or name == 'soft_crossentropy':
        return _C.LSoftCrossEntropy()
    else:
        NotImplementedError('Unknown loss')


def _get_metric(name):
    if name == 'mean_squared_error' or name == 'mse':
        return _C.MMeanSquaredError()
    elif name == 'categorical_accuracy' or name == 'accuracy':
        return _C.MCategoricalAccuracy()
    else:
        NotImplementedError('Unknown metric')

def _get_compserv(device):
    if device == 'cpu':
        return _C.EDDL.CS_CPU(4)
    elif device == 'gpu':
        NotImplementedError('GPU')
    elif device == 'fpga':
        NotImplementedError('FPGA')
    else:
        NotImplementedError('Unknown device')

def get_model(name='mlp'):
    if name == 'mlp':
        return _C.EDDL.get_model_mlp()
    elif name == 'cnn':
        return _C.EDDL.get_model_cnn()
    else:
        NotImplementedError('Unknown model')


def compile(model, optim, losses, metrics, device):
    optim = _get_optim(optim)
    losses = [_get_loss(l) for l in losses]
    metrics = [_get_metric(m) for m in metrics]
    compserv = _get_compserv(device)

    _C.EDDL.build(model, optim, losses, metrics, compserv)
    asdas = 3

def summary(model):
    return model.summary()


def plot(model, filename):
    return model.plot(filename)

def train_batch(x, y):
    pass
    #_C.train_batch(x, y)
