/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_REGULARIZER_H
#define EDDL_REGULARIZER_H

#include <string>

#include "../tensor/tensor.h"

using namespace std;


class Regularizer {
public:
    string name;
    // Todo: Implement
    explicit Regularizer(string name);
    virtual float set_weights(Tensor *T);
};


#endif //EDDL_REGULARIZER_H