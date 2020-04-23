/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LOSS_H
#define EDDL_LOSS_H

#include <cstdio>
#include <string>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"

using namespace std;

class Loss {
public:
    string name;

    explicit Loss(string name);

    virtual void delta(Tensor *T, Tensor *Y, Tensor *D);
    virtual float value(Tensor *T, Tensor *Y);
    virtual Loss* clone();
};


class LMeanSquaredError : public Loss {
public:
    LMeanSquaredError();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
    Loss* clone() override;
};


class LMeanAbsoluteError : public Loss {
public:
    LMeanAbsoluteError();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
    Loss* clone() override;
};


class LMeanRelativeError : public Loss {
public:
    LMeanRelativeError();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
    Loss* clone() override;
};


class LMeanSquaredLogarithmicError : public Loss {
public:
    LMeanSquaredLogarithmicError();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
    Loss* clone() override;
};


class LHinge : public Loss {
public:
    LHinge();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
    Loss* clone() override;
};


class LCrossEntropy : public Loss {
public:
    LCrossEntropy();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
    Loss* clone() override;
};

class LSoftCrossEntropy : public Loss {
public:
    LSoftCrossEntropy();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
    Loss* clone() override;
};

class LMin : public Loss {
public:
    LMin();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
    Loss* clone() override;
};


// TODO: Implement
//void mean_squared_error(Tensor *T, Tensor *Y, Tensor *D);
//void mean_absolute_error(Tensor *T, Tensor *Y, Tensor *D);
//void mean_absolute_percentage_error(Tensor *T, Tensor *Y, Tensor *D);
//void mean_squared_logarithmic_error(Tensor *T, Tensor *Y, Tensor *D);
//void squared_hinge(Tensor *T, Tensor *Y, Tensor *D);
//void hinge(Tensor *T, Tensor *Y, Tensor *D);
//void categorical_hinge(Tensor *T, Tensor *Y, Tensor *D);
//void logcosh(Tensor *T, Tensor *Y, Tensor *D);
//void categorical_crossentropy(Tensor *T, Tensor *Y, Tensor *D);
//void sparse_categorical_crossentropy(Tensor *T, Tensor *Y, Tensor *D);
//void binary_crossentropy(Tensor *T, Tensor *Y, Tensor *D);
//void kullback_leibler_divergence(Tensor *T, Tensor *Y, Tensor *D);
//void poisson(Tensor *T, Tensor *Y, Tensor *D);
//void cosine_proximity(Tensor *T, Tensor *Y, Tensor *D);

#endif
