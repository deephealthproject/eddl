Operators
=============

RNN
---------------

Fully-connected RNN where the output is to be fed back to input.

.. note::

    Not yet implemented

Example:

.. code-block:: c++
   :linenos:

    layer RNN(layer parent, int units, int num_layers, bool use_bias = true, float dropout = .0f, bool bidirectional = false, string name = "");


GRU
---------------

Gated Recurrent Unit - Cho et al. 2014.

.. note::

    Not yet implemented

Example:

.. code-block:: c++
   :linenos:

    layer GRU(layer parent, int units, int num_layers, bool use_bias = true, float dropout = .0f, bool bidirectional = false, string name = "");


LSTM
---------------

Long Short-Term Memory layer - Hochreiter 1997.

.. note::

    Not yet implemented

Example:

.. code-block:: c++
   :linenos:

    layer LSTM(layer parent, int units, int num_layers, bool use_bias = true, float dropout = .0f, bool bidirectional = false, string name = "");

