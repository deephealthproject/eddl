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

    l = GlorotNormal(l);


GlorotUniform
-------------

.. doxygenfunction:: GlorotUniform

Example:

.. code-block:: c++

    l = GlorotUniform(l);


RandomNormal
-------------

.. doxygenfunction:: RandomNormal

Example:

.. code-block:: c++

    l = RandomNormal(l, 0.0, 0.1);


RandomUniform
-------------

.. doxygenfunction:: RandomUniform

Example:

.. code-block:: c++

    l = RandomUniform(l, -0.05, 0.05);


Constant
-------------

.. doxygenfunction:: Constant

Example:

.. code-block:: c++

    l = Constant(l, 0.5);



HeUniform
-------------

.. doxygenfunction:: HeUniform

Example:

.. code-block:: c++

    l = HeUniform(l);


HeNormal
-------------

.. doxygenfunction:: HeNormal

Example:

.. code-block:: c++

    l = HeNormal(l);