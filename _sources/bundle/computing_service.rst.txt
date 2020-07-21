A computing service is an object that provides hardware transparency so you can easily change the hardware on which
the code will be executed.

CPU
====

.. doxygenfunction:: CS_CPU

Example:

.. code-block:: c++
   :linenos:

    compserv CS_CPU(int th=-1, string mem="low_mem");



GPU
====

.. doxygenfunction:: eddl::CS_GPU(const vector<int>, int, string)
.. doxygenfunction:: eddl::CS_GPU(const vector<int>, string)

Example:

.. code-block:: c++
   :linenos:

    compserv CS_GPU(const vector<int> g={1}, int lsb=1, string mem="low_mem");

    compserv CS_GPU(const vector<int> g={1}, string mem="low_mem");



FPGA
====
..
.. .. doxygenfunction:: eddl::CS_FGPA(const vector<int> &f, int lsb=1)

.. note::

    **Not implemented yet**

.. Example:
..
.. .. code-block:: c++
..    :linenos:
.. 
..     compserv CS_FGPA(const vector<int> &f, int lsb=1);



COMPSS
======

.. doxygenfunction:: CS_COMPSS

.. note::

    **Not implemented yet**


Example:

.. code-block:: c++
   :linenos:

    compserv CS_COMPSS(string filename);
