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
    virtual float apply(Tensor *T);
};

class RL1 : public Regularizer {
public:
    float l; // regularization factor

    explicit RL1(float l);
    float apply(Tensor *T) override;
};

class RL2 : public Regularizer {
public:
    float l; // regularization factor

    explicit RL2(float l);
    float apply(Tensor *T) override;
};

class RL1_L2 : public Regularizer {
public:
    float l1; // regularization factor for l1
    float l2; // regularization factor for l1

    explicit RL1_L2(float l1, float l2);
    float apply(Tensor *T) override;
};

#endif //EDDL_REGULARIZER_H