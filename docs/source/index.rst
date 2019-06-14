.. pyeddl documentation master file, created by
   sphinx-quickstart on Wed Jun  5 09:41:12 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyEddl's documentation!
==================================

.. figure:: ./_static/logo-pyeddl.png


PyEddl is a Python library that wraps the C++ European Distributed Deep Learning Library (EDDLL).
The code is open source, and `available on github`_.

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user/installation.rst
   user/faq.rst
   user/tutorial.rst


.. toctree::
   :maxdepth: 1
   :caption: Model

   models/model.rst


.. toctree::
   :maxdepth: 1
   :caption: Layers

   layers/conv.rst
   layers/core.rst
   layers/merge.rst
   layers/pool.rst


.. toctree::
   :maxdepth: 1
   :caption: Preprocessing

   preprocessing/losses.rst
   preprocessing/metrics.rst
   preprocessing/optimizers.rst
   preprocessing/datasets.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



.. _available on github: https://github.com/salvacarrion/pyeddl
