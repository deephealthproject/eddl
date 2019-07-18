
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

#ifndef EDDLL_LAYER_GENERATORS_H
#define EDDLL_LAYER_GENERATORS_H


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

    vector<Tensor *>input;

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

    LGauss(float mean, float stdev, initializer_list<int> &size, string name, int dev);
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

    LUniform(float low, float high, initializer_list<int> &size, string name, int dev);
    LUniform(float low, float high, vector<int> size, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};


#endif //EDDLL_LAYER_GENERATORS_H
