/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <iostream>
#include "tensor/tensor.h"
#include "../tests/dev/aux_tests.h"

using namespace std;

void pretty_res(string name, bool res){
    cout << "===================" << endl;
    cout << name << ": ";
    if(res){
        cout << "OK!";
    }else{
        cout << "FAILED!";
    }
    cout << endl;
    cout << "===================" << endl;
}

int main(int argc, char **argv) {
    pretty_res("MaxPool2D (CPU, correctness)", test_mpool());
//    pretty_res("MaxPool2D (GPU, correctness)", test_mpool());
//    pretty_res("MaxPool2D (CPU==GPU)", test_mpool());
}
