Optimizers (``pyeddl.optim``)
******************************************

.. automodule:: pyeddl.optim

The classes presented in this section are optimizers to modify the SGD updates
during the training of a model.

The update functions control the learning rate during the SGD optimization


.. autosummary::
    :nosignatures:

    SGD


Stochastic Gradient Descent
============================

This is the optimizer by default in all models.

.. autoclass:: SGD
   :members:
   :special-members: __init__
