/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_NOISE_H
#define EDDL_LAYER_NOISE_H


#include <string>
#include <cstdio>

#include "eddl/layers/layer.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;

/// GaussianNoise Layer
class LGaussianNoise : public LinLayer {
public:
    float stdev;
    static int total_layers;
    Tensor *noise;

    LGaussianNoise(Layer *parent, float stdev, string name, int dev, int mem);
    ~LGaussianNoise() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void mem_delta() override;
    void free_delta() override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};

#endif //EDDL_LAYER_NOISE_H
