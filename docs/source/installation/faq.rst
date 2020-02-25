FAQ
===


Is there a Python version?
--------------------------

Yes, the PyEDDL_ is the EDDL version for the Python lovers


Can I contribute?
------------------

Yes, but first open a new issue to explain and discuss your contribution.


Can I control the memory consumption?
-------------------------------------

Yes, we offer several memory levels to control the memory-speed trade-off. These levels are:


- ``full_mem`` (default): No memory bound (highest speed at the expense of the memory requirements)
- ``mid_mem``: Slight memory optimization (good trade-off memory-speed)
- ``low_mem``: Optimized for hardware with restricted memory capabilities.


Is it faster than PyTorch/TensorFlow/etc
----------------------------------------

Depends... Generally, there are many nuances so that a high-performance can be achieved (see benchmark section) and many of
those nuances are strongly tied to the classical memory-speed trade-off. To sum up, matrix multiplications on the CPU
are a 300% faster than their TensorFlow counterpart. Convolution times on GPU (no cuDNN) are slightly faster than on the
EDDL than PyTorch (210ms vs. 246ms), and using cuDNN  will get the cuDNN performance.


Is it more memory-efficient than PyTorch/TensorFlow/etc
-------------------------------------------------------

Depends on your memory setting (see the "Can I control the memory consumption?" answer).


.. _PyEDDL: https://github.com/deephealthproject/pyeddl