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

.. doxygenfunction:: Diff(layer l1, layer l2)

Example:

.. code-block:: c++
   :linenos:

   ...
   layer l = Diff(l1, l2); // l1 and l2 are layers with the same shape


.. doxygenfunction:: Diff(layer l1, float k)

Example:

.. code-block:: c++
   :linenos:

   ...
   l = Diff(l, 0.5);


.. doxygenfunction:: Diff(float k, layer l1)


Example:

.. code-block:: c++
   :linenos:

   ...
   l = Diff(0.5, l);
   


Division
---------------

Computes the division operation

.. doxygenfunction:: Div(layer l1, layer l2)

Example:

.. code-block:: c++
   :linenos:

   ...
   layer l = Div(l1, l2); // l1 and l2 are layers with the same shape



.. doxygenfunction:: Div(layer l1, float k)

Example:

.. code-block:: c++
   :linenos:

   ...
   l = Div(l, 0.5);



.. doxygenfunction:: Div(float k, layer l1)

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

.. doxygenfunction:: Mult(layer l1, layer l2)

Example:

.. code-block:: c++
   :linenos:

   ...
   layer l = Mult(l1, l2); // l1 and l2 are layers with the same shape
   

.. doxygenfunction:: Mult(layer l1, float k)

Example:

.. code-block:: c++
   :linenos:

   ...
   l = Mult(l, 2.0);


.. doxygenfunction:: Mult(float k, layer l1)

Example:

.. code-block:: c++
   :linenos:

   ...
   layer l = Mult(0.5, l);



Power
---------------

Computes the power operation

.. doxygenfunction:: Pow(layer l1, layer l2)

.. doxygenfunction:: Pow(layer l1, float k)





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

.. doxygenfunction:: Sum(layer l1, layer l2)

Example:

.. code-block:: c++
   :linenos:

   ...
   layer l = Sum(l1, l2); // l1 and l2 are layers with the same shape


.. doxygenfunction:: Sum(layer l1, float k)

Example:

.. code-block:: c++
   :linenos:

   ...
   l = Sum(l, 0.5);

.. doxygenfunction:: Sum(float k, layer l1)

Example:

.. code-block:: c++
   :linenos:

   ...
   l = Sum(0.5, l);


