Operators
=============

Abs
---------------

Computes the absolute operation

.. doxygenfunction:: eddl::Abs

Example:

.. code-block:: c++
   :linenos:

   ...
   l = Abs(l);




Subtraction
---------------

Computes the subtract operation

.. doxygenfunction:: eddl::Diff(layer, layer)

Example:

.. code-block:: c++
   :linenos:

   ...
   layer l = Diff(l1, l2); // l1 and l2 are layers with the same shape


.. doxygenfunction:: eddl::Diff(layer, float)

Example:

.. code-block:: c++
   :linenos:

   ...
   l = Diff(l, 0.5);


.. doxygenfunction:: eddl::Diff(float, layer)


Example:

.. code-block:: c++
   :linenos:

   ...
   l = Diff(0.5, l);
   


Division
---------------

Computes the division operation

.. doxygenfunction:: eddl::Div(layer, layer)

Example:

.. code-block:: c++
   :linenos:

   ...
   layer l = Div(l1, l2); // l1 and l2 are layers with the same shape



.. doxygenfunction:: eddl::Div(layer, float)

Example:

.. code-block:: c++
   :linenos:

   ...
   l = Div(l, 0.5);



.. doxygenfunction:: eddl::Div(float, layer)

Example:

.. code-block:: c++
   :linenos:

   ...
   l = Div(0.5, l);
   



Exponent
----------

Computes the exponent operation

.. doxygenfunction:: eddl::Exp



Example:

.. code-block:: c++
   :linenos:

   ...
   l = Exp(l);



Logarithm (natural)
-------------------

Computes the natural logarithm operation

.. doxygenfunction:: eddl::Log



Example:

.. code-block:: c++
   :linenos:

   ...
   l = Log(l);



Logarithm base 2
-----------------

Computes the logarithm of base 2 operation

.. doxygenfunction:: eddl::Log2



Example:

.. code-block:: c++
   :linenos:

   ...
   l = Log2(l);



Logarithm base 10
-----------------

Computes the logarithm of base 10 operation

.. doxygenfunction:: eddl::Log10



Example:

.. code-block:: c++
   :linenos:

   ...
   l = Log10(l);


Multiplication
---------------

Computes the product operation

.. doxygenfunction:: eddl::Mult(layer,layer)

Example:

.. code-block:: c++
   :linenos:

   ...
   layer l = Mult(l1, l2); // l1 and l2 are layers with the same shape
   

.. doxygenfunction:: eddl::Mult(layer,float)

Example:

.. code-block:: c++
   :linenos:

   ...
   l = Mult(l, 2.0);


.. doxygenfunction:: eddl::Mult(float,layer)

Example:

.. code-block:: c++
   :linenos:

   ...
   layer l = Mult(0.5, l);



Power
---------------

Computes the power operation

.. doxygenfunction:: eddl::Pow(layer,layer)

.. doxygenfunction:: eddl::Pow(layer,float)





Sqrt
---------------

Computes the power operation

.. doxygenfunction:: eddl::Sqrt



Example:

.. code-block:: c++
   :linenos:

   ...
   l = Sqrt(l);



Addition
---------------

Computes the power operation

.. doxygenfunction:: eddl::Sum(layer, layer)

Example:

.. code-block:: c++
   :linenos:

   ...
   layer l = Sum(l1, l2); // l1 and l2 are layers with the same shape


.. doxygenfunction:: eddl::Sum(layer, float)

Example:

.. code-block:: c++
   :linenos:

   ...
   l = Sum(l, 0.5);

.. doxygenfunction:: eddl::Sum(float, layer)

Example:

.. code-block:: c++
   :linenos:

   ...
   l = Sum(0.5, l);


