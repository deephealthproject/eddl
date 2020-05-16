/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_CONSTRAINT_H
#define EDDL_CONSTRAINT_H

#include <string>

#include "eddl/tensor/tensor.h"

class Constraint {
public:
    string name;
    // Todo: Implement
    explicit Constraint(string name);
    virtual float apply(Tensor *T);
};

class CMaxNorm : public Constraint {
public:
    float max_value;
    int axis;

    explicit CMaxNorm(float max_value, int axis);
    float apply(Tensor *T) override;
};

class CNonNeg : public Constraint {
public:

    explicit CNonNeg();
    float apply(Tensor *T) override;
};

class CUnitNorm : public Constraint {
public:
    int axis;

    explicit CUnitNorm(int axis);
    float apply(Tensor *T) override;
};

class CMinMaxNorm : public Constraint {
public:
    float min_value;
    float max_value;
    float rate;
    int axis;

    explicit CMinMaxNorm(float min_value, float max_value, float rate, int axis);
    float apply(Tensor *T) override;
};


#endif //EDDL_CONSTRAINT_H
