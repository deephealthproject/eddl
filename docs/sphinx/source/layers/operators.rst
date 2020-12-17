Operators
=============

Abs
---------------


.. doxygenfunction:: eddl::Abs

Example:

.. code-block:: c++

   l = Abs(l);




Subtraction
---------------

.. doxygenfunction:: Sub(layer l1, layer l2)

Example:

.. code-block:: c++

   layer l = Sub(l1, l2); // l1 and l2 are layers with the same shape



.. doxygenfunction:: Sub(layer l1, float k)

Example:

.. code-block:: c++

   l = Sub(l, 0.5);



.. doxygenfunction:: Sub(float k, layer l1)


Example:

.. code-block:: c++

   l = Sub(0.5, l);
   


Division
---------------

.. doxygenfunction:: Div(layer l1, layer l2)

Example:

.. code-block:: c++

   layer l = Div(l1, l2); // l1 and l2 are layers with the same shape



.. doxygenfunction:: Div(layer l1, float k)

Example:

.. code-block:: c++

   l = Div(l, 0.5);



.. doxygenfunction:: Div(float k, layer l1)

Example:

.. code-block:: c++

   l = Div(0.5, l);
   



Exponent
----------


.. doxygenfunction:: eddl::Exp


Example:

.. code-block:: c++

   l = Exp(l);



Logarithm (natural)
-------------------


.. doxygenfunction:: eddl::Log


Example:

.. code-block:: c++

   l = Log(l);



Logarithm base 2
-----------------


.. doxygenfunction:: eddl::Log2


Example:

.. code-block:: c++

   l = Log2(l);



Logarithm base 10
-----------------


.. doxygenfunction:: eddl::Log10



Example:

.. code-block:: c++

   l = Log10(l);



Multiplication
---------------


.. doxygenfunction:: Mult(layer l1, layer l2)

Example:

.. code-block:: c++

   layer l = Mult(l1, l2); // l1 and l2 are layers with the same shape
   


.. doxygenfunction:: Mult(layer l1, float k)

Example:

.. code-block:: c++

   l = Mult(l, 2.0);



.. doxygenfunction:: Mult(float k, layer l1)

Example:

.. code-block:: c++

   layer l = Mult(0.5, l);



Power
---------------


.. doxygenfunction:: Pow(layer l1, layer l2)

.. doxygenfunction:: Pow(layer l1, float k)





Sqrt
---------------


.. doxygenfunction:: eddl::Sqrt



Example:

.. code-block:: c++

   l = Sqrt(l);



Addition
---------------


.. doxygenfunction:: Add(layer l1, layer l2)

Example:

.. code-block:: c++

   layer l = Add(l1, l2); // l1 and l2 are layers with the same shape



.. doxygenfunction:: Add(layer l1, float k)

Example:

.. code-block:: c++

   l = Add(l, 0.5);



.. doxygenfunction:: Add(float k, layer l1)

Example:

.. code-block:: c++

   l = Add(0.5, l);


