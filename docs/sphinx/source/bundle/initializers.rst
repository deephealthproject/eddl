Initializers
============

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/docs/markdown/eddl_progress.md#initializers

GlorotNormal
------------

.. doxygenfunction:: GlorotNormal

Example:

.. code-block:: c++
   :linenos:

    layer GlorotNormal(layer l,int seed=1234);


GlorotUniform
-------------

.. doxygenfunction:: GlorotUniform

Example:

.. code-block:: c++
   :linenos:

    layer GlorotUniform(layer l,int seed=1234);


RandomNormal
-------------

.. doxygenfunction:: RandomNormal

Example:

.. code-block:: c++
   :linenos:

    layer RandomNormal(layer l, float m=0.0,float s=0.1, float seed=1234);


RandomUniform
-------------

.. doxygenfunction:: RandomUniform

Example:

.. code-block:: c++
   :linenos:

    layer RandomUniform(layer l, float min=0.0,float max=0.1, float seed=1234);

HeUniform
-------------

.. doxygenfunction:: HeUniform

Example:

.. code-block:: c++
   :linenos:

    layer HeUniform(layer l,int seed=1234);


Constant
-------------

.. doxygenfunction:: Constant

Example:

.. code-block:: c++
   :linenos:

    layer Constant(layer l, float v=0.1);
