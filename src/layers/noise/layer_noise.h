
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

#ifndef EDDLL_LAYER_NOISE_H
#define EDDLL_LAYER_NOISE_H


#include <string>
#include <stdio.h>

#include "../layer.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;

/// GaussianNoise Layer
class LGaussianNoise : public LinLayer {
public:
    float stdev;
    static int total_layers;
    Tensor *noise;

    LGaussianNoise(Layer *parent, float stdev, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;
    
    void forward() override;

    void backward() override;

    string plot(int c) override;
};

#endif //EDDLL_LAYER_NOISE_H
