First example
-------------

.. raw:: html

    <div style="position: relative; padding-bottom: 3em; text-align: center">
    <iframe width="560" height="315" src="https://www.youtube.com/embed/yZtYnqbcnSo" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>


.. code:: c++

    #include <iostream>

    #include <eddl/tensor/tensor.h>

    int main(int argc, char* argv[]){
        Tensor* t = Tensor::ones({5, 5, 5});
        std::cout << "Tensor sum=" << t->sum() << res;

        return 0;
    }

This example simply sums all the elements of a tensor
