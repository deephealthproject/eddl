/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_GENERATORS_H
#define EDDL_LAYER_GENERATORS_H


#include <string>
#include <stdio.h>

#include "../layer.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;


/////////////////////////////////////////
/////////////////////////////////////////
// Operator layer
class GeneratorLayer : public Layer {
public:

    int binary;
    float val;

    GeneratorLayer(string name, int dev);

    void addchild(Layer *l) override;

    void addparent(Layer *l) override;


};

/// Gaussian Layer
class LGauss : public GeneratorLayer {
public:
    static int total_layers;
    vector<int> size;
    float mean;
    float stdev;

    Tensor *mask;

    LGauss(float mean, float stdev, vector<int> size, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Uniform Layer
class LUniform : public GeneratorLayer {
public:
    static int total_layers;
    vector<int> size;
    float low;
    float high;

    Tensor *mask;

    LUniform(float low, float high, vector<int> size, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};


#endif //EDDL_LAYER_GENERATORS_H
