/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_INITIALIZER_H
#define EDDL_INITIALIZER_H

#include <string>

#include "eddl/tensor/tensor.h"

using namespace std;


class Initializer {
public:
    string name;

    explicit Initializer(string name);
    ~Initializer();
    virtual void apply(Tensor *params) = 0;  // Pure virtual
};

class IConstant : public Initializer {
public:
    float value;

    explicit IConstant(float value);
    void apply(Tensor *params) override;
};

class IIdentity : public Initializer {
public:
    float gain;

    explicit IIdentity(float gain);
    void apply(Tensor *params) override;
};

class IGlorotNormal : public Initializer {
public:
    int seed;

    explicit IGlorotNormal(int seed=-1);
    void apply(Tensor *params) override;
};

class IGlorotUniform : public Initializer {
public:
    int seed;

    explicit IGlorotUniform(int seed=-1);
    void apply(Tensor *params) override;
};

class IHeNormal : public Initializer {
public:
    int seed;

    explicit IHeNormal(int seed=-1);
    void apply(Tensor *params) override;
};

class IHeUniform : public Initializer {
public:
    int seed;

    explicit IHeUniform(int seed=-1);
    void apply(Tensor *params) override;
};


class IRandomNormal : public Initializer {
public:
    float mean;
    float stdev;
    int seed;

    explicit IRandomNormal(float mean, float stdev, int seed=-1);
    void apply(Tensor *params) override;
};

class IRandomUniform : public Initializer {
public:
    float minval;
    float maxval;
    int seed;

    explicit IRandomUniform(float minval, float maxval, int seed=-1);
    void apply(Tensor *params) override;
};

class IOrthogonal : public Initializer {
public:
    float gain;
    int seed;

    explicit IOrthogonal(float gain, int seed=-1);
    void apply(Tensor *params) override;
};

class ITruncateNormal : public Initializer {
public:
    float mean;
    float stdev;
    int seed;

    explicit ITruncateNormal(float mean, float stdev, int seed=-1);
    void apply(Tensor *params) override;
};

class IVarianceScaling : public Initializer {
public:
    float scale;
    string mode;
    string distribution;
    int seed;

    explicit IVarianceScaling(float scale, string mode, string distribution, int seed=-1);
    void apply(Tensor *params) override;
};

#endif //EDDL_INITIALIZER_H
