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


<<<<<<< HEAD
=======
Is it faster than PyTorch/TensorFlow/etc
----------------------------------------

Check our benchmakrs: eddl_benchmarks_


Is it more memory-efficient than PyTorch/TensorFlow/etc
-------------------------------------------------------

Depends on your memory setting (see the "Can I control the memory consumption?" answer).

>>>>>>> 7e92b09a7beb637dec4e386730994a7f2c712062

.. _PyEDDL: https://github.com/deephealthproject/pyeddl
.. _eddl_benchmarks: https://github.com/jofuelo/eddl_benchmark