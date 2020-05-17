/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_METRIC_H
#define EDDL_METRIC_H

#include <cstdio>
#include <string>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"


using namespace std;

class Metric {
public:
    string name;

    explicit Metric(string name);

    virtual float value(Tensor *T, Tensor *Y);
    virtual Metric* clone();
};

class MDice : public Metric {
public:
    MDice();

    float value(Tensor *T, Tensor *Y) override;
    Metric* clone() override;
};

class MMeanSquaredError : public Metric {
public:
    MMeanSquaredError();

    float value(Tensor *T, Tensor *Y) override;
    Metric* clone() override;
};

class MMeanRelativeError : public Metric {
public:
    MMeanRelativeError(float eps=0.001);
    float eps;

    float value(Tensor *T, Tensor *Y) override;
    Metric* clone() override;
};

class MMeanAbsoluteError : public Metric {
public:
    MMeanAbsoluteError();

    float value(Tensor *T, Tensor *Y) override;
    Metric* clone() override;
};



class MCategoricalAccuracy : public Metric {
public:
    MCategoricalAccuracy();

    float value(Tensor *T, Tensor *Y) override;
    Metric* clone() override;
};

class MSum : public Metric {
public:
    MSum();

    float value(Tensor *T, Tensor *Y) override;
    Metric* clone() override;
};


#endif
