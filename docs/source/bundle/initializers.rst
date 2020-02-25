Initializers
============

GlorotNormal
------------

Glorot normal initializer, also called Xavier normal initializer.

Example:

.. code-block:: c++
   :linenos:

    layer GlorotNormal(layer l,int seed=1234);


GlorotUniform
-------------

Glorot uniform initializer, also called Xavier uniform initializer.

Example:

.. code-block:: c++
   :linenos:

    layer GlorotUniform(layer l,int seed=1234);


RandomNormal
-------------

Random normal initializer.

Example:

.. code-block:: c++
   :linenos:

    layer RandomNormal(layer l, float m=0.0,float s=0.1, float seed=1234);


RandomUniform
-------------

Random uniform initializer.

Example:

.. code-block:: c++
   :linenos:

    layer RandomUniform(layer l, float min=0.0,float max=0.1, float seed=1234);


Constant
-------------

Initializer that generates tensors initialized to a constant value.
Example:

.. code-block:: c++
   :linenos:

    layer Constant(layer l, float v=0.1);
