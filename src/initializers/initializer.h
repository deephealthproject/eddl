/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_INITIALIZER_H
#define EDDL_INITIALIZER_H

#include <string>

#include "../tensor/tensor.h"

using namespace std;


class Initializer {
public:
    string name;
    // Todo: Implement
    explicit Initializer(string name);
    virtual float set_weights(Tensor *T);
};

class IConstant : public Initializer {
public:
    float value;

    explicit IConstant(float value);
    float set_weights(Tensor *T) override;
};

class IIdentity : public Initializer {
public:
    float gain;

    explicit IIdentity(float gain);
    float set_weights(Tensor *T) override;
};

class IGlorotNormal : public Initializer {
public:
    int seed;

    explicit IGlorotNormal(int seed=-1);
    float set_weights(Tensor *T) override;
};

class IGlorotUniform : public Initializer {
public:
    int seed;

    explicit IGlorotUniform(int seed=-1);
    float set_weights(Tensor *T) override;
};

class IRandomNormal : public Initializer {
public:
    float mean;
    float stdev;
    int seed;

    explicit IRandomNormal(float mean, float stdev, int seed=-1);
    float set_weights(Tensor *T) override;
};

class IRandomUniform : public Initializer {
public:
    float minval;
    float maxval;
    int seed;

    explicit IRandomUniform(float minval, float maxval, int seed=-1);
    float set_weights(Tensor *T) override;
};

class IOrthogonal : public Initializer {
public:
    float gain;
    int seed;

    explicit IOrthogonal(float gain, int seed=-1);
    float set_weights(Tensor *T) override;
};

#endif //EDDL_INITIALIZER_H
