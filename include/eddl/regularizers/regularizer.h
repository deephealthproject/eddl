/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_REGULARIZER_H
#define EDDL_REGULARIZER_H

#include <string>

#include "eddl/tensor/tensor.h"

using namespace std;


class Regularizer {
public:
    string name;
    // Todo: Implement
    explicit Regularizer(string name);
    virtual void apply(Tensor *T) = 0;
};

class RL1 : public Regularizer {
public:
    float l1; // regularization factor

    explicit RL1(float l1);
    void apply(Tensor *T) override;
};

class RL2 : public Regularizer {
public:
    float l2; // regularization factor

    explicit RL2(float l2);
    void apply(Tensor *T) override;
};

class RL1L2 : public Regularizer {
public:
    float l1;
    float l2;

    explicit RL1L2(float l1, float l2);
    void apply(Tensor *T) override;
};

#endif //EDDL_REGULARIZER_H
