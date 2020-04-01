/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef _LOSS_
#define _LOSS_

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
};


class LMeanSquaredError : public Loss {
public:
    LMeanSquaredError();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
};


class LMeanAbsoluteError : public Loss {
public:
    LMeanAbsoluteError();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
};


class LMeanRelativeError : public Loss {
public:
    LMeanRelativeError();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
};


class LMeanSquaredLogarithmicError : public Loss {
public:
    LMeanSquaredLogarithmicError();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
};


class LHinge : public Loss {
public:
    LHinge();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
};


class LCrossEntropy : public Loss {
public:
    LCrossEntropy();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
};

class LSoftCrossEntropy : public Loss {
public:
    LSoftCrossEntropy();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
};

class LMin : public Loss {
public:
    LMin();

    void delta(Tensor *T, Tensor *Y, Tensor *D) override;
    float value(Tensor *T, Tensor *Y) override;
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
