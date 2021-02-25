FAQ
===


Python library
---------------

Is there a Python version?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes, the PyEDDL_ is the EDDL version for the Python lovers.


Is the PyEDDL develop by the same team?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sort of. We are all part of the DeepHealth project and, although each team is specialized
in developing and supporting different libraries, we all contribute to the other projects so that the whole DeepHealth
ecosystem can work smoothly.


Contributions
---------------

Can I contribute?
^^^^^^^^^^^^^^^^^^

Yes, but first you need to open a new issue to explain and discuss your contribution.


Installation & Support
-------------------------

Do you have X feature supported?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have two markdown where you can check the layers and operations that we currently support:

- `Layers supported <https://github.com/deephealthproject/eddl/blob/master/docs/markdown/eddl_progress.md>`_
- `Tensor operations supported <https://github.com/deephealthproject/eddl/blob/master/docs/markdown/eddl_progress_tensor.md>`_

I need X feature, can you add it?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Maybe... Open a new `issue`_ and we will happily discuss it.

Do you have a Dockerfile prepared?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes, it's on the root directory of the repo.

.. code:: bash

    # Build the image
    cd eddl/
    docker build -f Dockerfile .

    # Enter in the terminal
    docker exec -it [container-id] bash


Performance
---------------

Is it faster than PyTorch/TensorFlow/etc?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Check our benchmakrs: `EDDL benchmarks`_

**Summary:**

- 50% faster than PyTorch
- Similar performance as Keras


Is it more memory-efficient than PyTorch/TensorFlow/etc?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Depends on your memory setting. 
You can take a look at our benchmarks: `EDDL benchmarks`_

.. _PyEDDL: https://github.com/deephealthproject/pyeddl
.. _`EDDL benchmarks`: https://github.com/jofuelo/eddl_benchmark
.. _`issue`: https://github.com/deephealthproject/eddl/issues
