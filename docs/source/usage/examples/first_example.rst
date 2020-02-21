First example
-------------

Copy this code to a new file, and let's name it ``main.cpp``:

.. code:: c++

    #include <iostream>

    #include <eddl/tensor/tensor.h>

    int main(int argc, char* argv[]){
        Tensor* t = Tensor::ones({5, 5, 5});
        std::cout << "Tensor sum=" << t->sum() << res;

        return 0;
    }

This example simply sums all the elements of a tensor

To compile it we are going to use CMake and the ``find_package(eddl)`` function.
If you are not familiar with CMake, read the next section.