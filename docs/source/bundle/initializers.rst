Initializers
============

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/eddl_progress.md#initializers

GlorotNormal
------------

Glorot normal initializer, also called Xavier normal initializer.



  Parameters:

  - ``l``: The layer whose weights must be initialized.
  - ``seed``: An integer. Used to seed the random generator.

Example:

.. code-block:: c++
   :linenos:

    layer GlorotNormal(layer l,int seed=1234);


GlorotUniform
-------------

Glorot uniform initializer, also called Xavier uniform initializer.

  Parameters:

  - ``l``: The layer whose weights must be initialized.
  - ``seed``: An integer. Used to seed the random generator.

Example:

.. code-block:: c++
   :linenos:

    layer GlorotUniform(layer l,int seed=1234);


RandomNormal
-------------

Random normal initializer.

  Parameters:

  - ``l``: The layer whose weights must be initialized.
  - ``m``: A float. Mean of the random values to generate.
  - ``s``: A float. Standard deviation of the random values to generate.
  - ``seed``: An integer. Used to seed the random generator.

Example:

.. code-block:: c++
   :linenos:

    layer RandomNormal(layer l, float m=0.0,float s=0.1, float seed=1234);


RandomUniform
-------------

Random uniform initializer.

  Parameters:

    - ``l``: The layer whose weights must be initialized.
    - ``min``: A float. Lower bound of the range of random  values to generate.
    - ``max``: A float. Upper bound of the range of random values to generate.
    - ``seed``: An integer. Used to seed the random generator.

Example:

.. code-block:: c++
   :linenos:

    layer RandomUniform(layer l, float min=0.0,float max=0.1, float seed=1234);


Constant
-------------

Initializer that generates tensors initialized to a constant value.

  Parameters:

    - ``l``: The layer whose weights must be initialized.
    - ``v``: A float. The constant value to initialize the tensor.
Example:

.. code-block:: c++
   :linenos:

    layer Constant(layer l, float v=0.1);
