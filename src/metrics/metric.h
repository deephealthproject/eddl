/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef _METRIC_
#define _METRIC_

#include <stdio.h>
#include <string>

#include "../tensor/tensor.h"
#include "../tensor/nn/tensor_nn.h"


using namespace std;

class Metric {
public:
    string name;

    explicit Metric(string name);

    virtual float value(Tensor *T, Tensor *Y);
};


class MMeanSquaredError : public Metric {
public:
    MMeanSquaredError();

    float value(Tensor *T, Tensor *Y) override;
};

class MMeanRelativeError : public Metric {
public:
    MMeanRelativeError(float eps=0.001);
    float eps;

    float value(Tensor *T, Tensor *Y) override;
};

class MMeanAbsoluteError : public Metric {
public:
    MMeanAbsoluteError();

    float value(Tensor *T, Tensor *Y) override;
};



class MCategoricalAccuracy : public Metric {
public:
    MCategoricalAccuracy();

    float value(Tensor *T, Tensor *Y) override;
};

class MSum : public Metric {
public:
    MSum();

    float value(Tensor *T, Tensor *Y) override;
};


#endif
