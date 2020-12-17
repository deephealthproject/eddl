Merge
=====

Add
----

.. doxygenfunction:: Add(layer l1, layer l2)

Example:

.. code-block:: c++

   layer in1 = Input({3,584,584});
   layer in2 = Input({3,584,584});

   layer l = Add({in1,in2});
   



Average
-------

.. doxygenfunction:: Average

Example:

.. code-block:: c++

   layer in1 = Input({3,584,584});
   layer in2 = Input({3,584,584});

   layer l = Average({in1,in2});
   


Concat
------

.. doxygenfunction:: Concat

Example:

.. code-block:: c++

   layer in1 = Input({3,584,584});
   layer in2 = Input({1,584,584});

   layer l = Concat({in1,in2});
   


MatMul
------

.. doxygenfunction:: MatMul


Example:

.. code-block:: c++

   layer in1 = Input({3,584,584});
   layer in2 = Input({3,584,584});

   layer l = MatMul({in1,in2});
   


Maximum
-------

.. doxygenfunction:: Maximum

Example:

.. code-block:: c++

   layer in1 = Input({3,584,584});
   layer in2 = Input({3,584,584});

   layer l = Maximum({in1,in2});
   


Minimum
-------

.. doxygenfunction:: Minimum

Example:

.. code-block:: c++

   layer in1 = Input({3,584,584});
   layer in2 = Input({3,584,584});

   layer l = Minimum({in1,in2});
   


Subtract
---------

.. doxygenfunction:: Subtract

Example:

.. code-block:: c++

   layer in1 = Input({3,584,584});
   layer in2 = Input({3,584,584});

   layer l = Subtract({in1,in2});
   

