Generators
================

Gaussian Generator
------------------

.. doxygenfunction:: GaussGenerator



Example:

.. code-block:: c++

   layer gin = GaussGenerator(0.0, 1, {3, 32, 32});



Uniform Generator
-----------------

.. doxygenfunction:: UniformGenerator


.. code-block:: c++

   layer gin = UniformGenerator(3.0, 10.0, {3, 32, 32});
