Recurrent
=============

RNN
---------------

.. doxygenfunction:: RNN

Example:

.. code-block:: c++

    l = RNN(lE,32);

You can check a full example on :doc:`../usage/advanced`.


GRU
---------------

Gated Recurrent Unit - Cho et al. 2014.

.. doxygenfunction:: GRU(layer parent, int units, bool mask_zeros=false, bool bidirectional = false, string name = "")


Example:

.. code-block:: c++

    l = GRU(l, 128);



LSTM
---------------

.. doxygenfunction:: LSTM(layer parent, int units, bool mask_zeros = false, bool bidirectional = false, string name = "")

.. doxygenfunction:: LSTM(vector<layer> parent, int units, bool mask_zeros, bool bidirectional, string name)

Example:

.. code-block:: c++

    l = LSTM(lE,32);

You can check a full example on :doc:`../usage/advanced`.