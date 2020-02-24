Operators
=============

Abs
---------------

Computes the absolute operation

Example:

.. code-block:: c++
   :linenos:

    layer Abs(layer l);



Subtraction
---------------

Computes the subtract operation

Example:

.. code-block:: c++
   :linenos:

    layer Diff(layer l1, layer l2);
    layer Diff(layer l1, float k);
    layer Diff(float k, layer l1);



Division
---------------

Computes the division operation

Example:

.. code-block:: c++
   :linenos:

    layer Div(layer l1, layer l2);
    layer Div(layer l1, float k);
    layer Div(float k, layer l1);


Exponent
----------

Computes the exponent operation

Example:

.. code-block:: c++
   :linenos:

    layer Exp(layer l);



Logarithm (natural)
-------------------

Computes the natural logarithm operation

Example:

.. code-block:: c++
   :linenos:

    layer Log(layer l);



Logarithm base 2
---------------

Computes the logarithm of base 2 operation

Example:

.. code-block:: c++
   :linenos:

    layer Log2(layer l);



Logarithm base 10
-----------------

Computes the logarithm of base 10 operation

Example:

.. code-block:: c++
   :linenos:

    layer Abs(layer l);



Multiplication
---------------

Computes the product operation

Example:

.. code-block:: c++
   :linenos:

    layer Mult(layer l1, layer l2);
    layer Mult(layer l1, float k);
    layer Mult(float k,layer l1);



Power
---------------

Computes the power operation

Example:

.. code-block:: c++
   :linenos:

    layer Pow(layer l1, layer l2);
    layer Pow(layer l1, float k);



Sqrt
---------------

Computes the power operation

Example:

.. code-block:: c++
   :linenos:

    layer Sqrt(layer l);



Addition
---------------

Computes the power operation

Example:

.. code-block:: c++
   :linenos:

    layer Sum(layer l1, layer l2);
    layer Sum(layer l1, float k);
    layer Sum(float k, layer l1);

Select
---------------

Selects a subset of the output tensor using indices (similar to Numpy; the batch is ignored)

Example:

.. code-block:: c++
   :linenos:

    layer Select(layer l, vector<string> indices, string name="");
    // e.g.: Select(l, {"-1", "20:100", "50:-10", ":"}



Permute
---------------

Permute the axis of the output tensor (the batch is ignored)

Example:

.. code-block:: c++
   :linenos:

    layer Permute(layer l, vector<int> dims, string name="");
    // e.g.: Permute(l, {0, 3, 1, 2})




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
