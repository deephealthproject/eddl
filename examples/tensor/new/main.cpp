#include <iostream>
#include <eddl/tensor/tensor.h>

int main(){
    Tensor *t1 = Tensor::ones({5, 1});
    std::cout << t1->sum() << std::endl;
}
