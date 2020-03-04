Operators
=============

Abs
---------------

Computes the absolute operation

.. doxygenfunction:: eddl::Abs

Example:

.. code-block:: c++
   :linenos:

    layer Abs(layer l);



Subtraction
---------------

Computes the subtract operation

.. doxygenfunction:: eddl::Diff(layer, layer)


.. doxygenfunction:: eddl::Diff(layer, float)


.. doxygenfunction:: eddl::Diff(float, layer)


Example:

.. code-block:: c++
   :linenos:

    layer Diff(layer l1, layer l2);
    layer Diff(layer l1, float k);
    layer Diff(float k, layer l1);



Division
---------------

Computes the division operation

.. doxygenfunction:: eddl::Div(layer, layer)

.. doxygenfunction:: eddl::Div(layer, float)

.. doxygenfunction:: eddl::Div(float, layer)


Example:

.. code-block:: c++
   :linenos:

    layer Div(layer l1, layer l2);
    layer Div(layer l1, float k);
    layer Div(float k, layer l1);


Exponent
----------

Computes the exponent operation

.. doxygenfunction:: eddl::Exp



Example:

.. code-block:: c++
   :linenos:

    layer Exp(layer l);



Logarithm (natural)
-------------------

Computes the natural logarithm operation

.. doxygenfunction:: eddl::Log



Example:

.. code-block:: c++
   :linenos:

    layer Log(layer l);



Logarithm base 2
-----------------

Computes the logarithm of base 2 operation

.. doxygenfunction:: eddl::Log2



Example:

.. code-block:: c++
   :linenos:

    layer Log2(layer l);



Logarithm base 10
-----------------

Computes the logarithm of base 10 operation

.. doxygenfunction:: eddl::Log10



Example:

.. code-block:: c++
   :linenos:

    layer Abs(layer l);



Multiplication
---------------

Computes the product operation

.. doxygenfunction:: eddl::Mult(layer,layer)

.. doxygenfunction:: eddl::Mult(layer,float)

.. doxygenfunction:: eddl::Mult(float,layer)



Example:

.. code-block:: c++
   :linenos:

    layer Mult(layer l1, layer l2);
    layer Mult(layer l1, float k);
    layer Mult(float k,layer l1);



Power
---------------

Computes the power operation

.. doxygenfunction:: eddl::Pow(layer,layer)

.. doxygenfunction:: eddl::Pow(layer,float)




Example:

.. code-block:: c++
   :linenos:

    layer Pow(layer l1, layer l2);
    layer Pow(layer l1, float k);



Sqrt
---------------

Computes the power operation

.. doxygenfunction:: eddl::Sqrt



Example:

.. code-block:: c++
   :linenos:

    layer Sqrt(layer l);



Addition
---------------

Computes the power operation

.. doxygenfunction:: eddl::Sum(layer, layer)

.. doxygenfunction:: eddl::Sum(layer, float)

.. doxygenfunction:: eddl::Sum(float, layer)



Example:

.. code-block:: c++
   :linenos:

    layer Sum(layer l1, layer l2);
    layer Sum(layer l1, float k);
    layer Sum(float k, layer l1);

Select
---------------

Selects a subset of the output tensor using indices (similar to Numpy; the batch is ignored)


.. doxygenfunction:: eddl::Select


Example:

.. code-block:: c++
   :linenos:

    layer Select(layer l, vector<string> indices, string name="");
    // e.g.: Select(l, {"-1", "20:100", "50:-10", ":"}



Permute
---------------

Permute the axis of the output tensor (the batch is ignored)

.. doxygenfunction:: eddl::Permute



Example:

.. code-block:: c++
   :linenos:

    layer Permute(layer l, vector<int> dims, string name="");
    // e.g.: Permute(l, {0, 3, 1, 2})

