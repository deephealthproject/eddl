Classification
==============

MNIST
------

MNIST database of handwritten digits

Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.

.. doxygenfunction:: eddl::download_mnist

Example:

.. code-block:: c++
   :linenos:

    void download_mnist();


CIFAR-10
--------

CIFAR10 small image classification

Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.

.. doxygenfunction:: eddl::download_cifar10

Example:

.. code-block:: c++
   :linenos:

    void download_cifar10();
