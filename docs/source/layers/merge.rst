Merge
=====

Add
----

.. doxygenfunction:: Add

Example:

.. code-block:: c++
   :linenos:

    layer Add(const vector<layer> &layers, string name = "");



Average
-------

.. doxygenfunction:: Average

Example:

.. code-block:: c++
   :linenos:

    layer Average(const vector<layer> &layers, string name = ""); //Todo: Implement



Concat
------

.. doxygenfunction:: Concat

Example:

.. code-block:: c++
   :linenos:

    layer Concat(const vector<layer> &layers, unsigned int axis=1, string name = "");



MatMul
------

It takes as input a list of layers, all of the same shape, and returns a single tensor (also of the same shape).

Example:

.. code-block:: c++
   :linenos:

    layer MatMul(const vector<layer> &layers, string name = "");



Maximum
-------

.. doxygenfunction:: Maximum

Example:

.. code-block:: c++
   :linenos:

    layer Maximum(const vector<layer> &layers, string name = "");



Minimum
-------

.. doxygenfunction:: Minimum

Example:

.. code-block:: c++
   :linenos:

    layer Minimum(const vector<layer> &layers, string name = "");



Subtract
---------

.. doxygenfunction:: Subtract

Example:

.. code-block:: c++
   :linenos:

    layer Subtract(const vector<layer> &layers, string name = "");


