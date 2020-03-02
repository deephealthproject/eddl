Reduction Layers
================


ReduceMean
---------------

Reduced mean

Example:

.. code-block:: c++
   :linenos:

    layer ReduceMean(layer l, vector<int> axis = {0}, bool keepdims = false);


ReduceVar
---------------

Reduced var

Example:

.. code-block:: c++
   :linenos:

    layer ReduceVar(layer l, vector<int> axis = {0}, bool keepdims = false);


ReduceSum
---------------

Reduced sum

Example:

.. code-block:: c++
   :linenos:

    layer ReduceSum(layer l, vector<int> axis = {0}, bool keepdims = false);


ReduceMax
---------------

Reduced max

Example:

.. code-block:: c++
   :linenos:

    layer ReduceMax(layer l, vector<int> axis = {0}, bool keepdims = false);


ReduceMin
---------------

Reduced min

Example:

.. code-block:: c++
   :linenos:

    layer ReduceMin(layer l, vector<int> axis = {0}, bool keepdims = false);
