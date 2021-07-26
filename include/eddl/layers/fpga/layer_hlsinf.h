/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_HLSINF_H
#define EDDL_LAYER_HLSINF_H


#include <string>
#include <cstdio>

#include "eddl/layers/layer.h"
#include "eddl/layers/merge/layer_merge.h"

using namespace std;

/// HLSinf Layer
class LHLSinf : public LinLayer {
public:

    int KH, KW, SH, SW, PH, PW;  // convolution
    int enable_relu;    
    int enable_maxpooling;
    int enable_add;
    int enable_stm;

    Tensor *filter;
    Tensor *bias;

    static int total_layers;

    LHLSinf(Layer *parent, string name, int dev, int mem, int KH, int KW, int SH, int SW, int PH, int PW, int enable_relu, int enable_maxpooling, int enable_add, int enable_stm);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};

#endif //EDDL_LAYER_HLSINF_H
