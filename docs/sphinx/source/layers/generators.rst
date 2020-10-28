Generators
================

Gaussian Generator
------------------

.. doxygenfunction:: GaussGenerator

Generates a gaussian noise output (typically used for GANs)

Example:

.. code-block:: c++

   layer gin = GaussGenerator(0.0, 1, {3, 32, 32});



Uniform Generator
-----------------

.. doxygenfunction:: UniformGenerator

Generates a uniform noise output (typically used for GANs)



