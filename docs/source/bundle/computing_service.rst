A computing service is an object that provides hardware transparency so you can easily change the hardware on which
the code will be executed.

CPU
====

Executes the code in the CPU.

- ``th``: Indicates the number of threads to use (``-1 = all available threads``)
- ``mem``: Indicates the memory consumption of the model

    - ``full_mem``: (default): No memory bound (highest speed at the expense of the memory requirements)
    - ``mid_mem``: Slight memory optimization (good trade-off memory-speed)
    - ``low_mem``: Optimized for hardware with restricted memory capabilities.

Example:

.. code-block:: c++
   :linenos:

    compserv CS_CPU(int th=-1, string mem="low_mem");



GPU
====

Executes the code in the GPU.

- ``g``: Vector of bools to set which GPUs will be use (``1=on, 0=off``)
- ``lsb``: (Multi-gpu setting) Number of batches to run before synchronizing the weighs of the different GPUs
- ``mem``: Indicates the memory consumption of the model

    - ``full_mem``: (default): No memory bound (highest speed at the expense of the memory requirements)
    - ``mid_mem``: Slight memory optimization (good trade-off memory-speed)
    - ``low_mem``: Optimized for hardware with restricted memory capabilities.

Example:

.. code-block:: c++
   :linenos:

    compserv CS_GPU(const vector<int> g={1}, int lsb=1, string mem="low_mem");



FPGA
====

Executes the code in the FPGA.

- ``f``: Vector of bools to set which FPGAs will be use (``1=on, 0=off``)
- ``lsb``: (Multi-gpu setting) Number of batches to run before synchronizing the weighs of the different GPUs

.. note::

    Not yet implemented

Example:

.. code-block:: c++
   :linenos:

    compserv CS_FGPA(const vector<int> &f, int lsb=1);



COMPSS
======

Executes the code through the COMP Superscalar (COMPSs) framework

- ``filename``: File with the setup specification

.. note::

    Not yet implemented

Example:

.. code-block:: c++
   :linenos:

    compserv CS_COMPSS(string filename);
