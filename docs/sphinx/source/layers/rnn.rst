Recurrent
=============

RNN
---------------

.. doxygenfunction:: RNN

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#core-layers

Example:

.. code-block:: c++
   :linenos:

    layer RNN(layer parent, int units, int num_layers, bool use_bias = true, float dropout = .0f, bool bidirectional = false, string name = "");


GRU
---------------

Gated Recurrent Unit - Cho et al. 2014.

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#core-layers

Example:

.. code-block:: c++
   :linenos:

    layer GRU(layer parent, int units, int num_layers, bool use_bias = true, float dropout = .0f, bool bidirectional = false, string name = "");


LSTM
---------------

.. doxygenfunction:: LSTM

.. note::

    **Not implemented yet**

    Check development progress in https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#core-layers

Example:

.. code-block:: c++
   :linenos:

    layer LSTM(layer parent, int units, int num_layers, bool use_bias = true, float dropout = .0f, bool bidirectional = false, string name = "");

