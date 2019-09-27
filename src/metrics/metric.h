
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


class MCategoricalAccuracy : public Metric {
public:
    MCategoricalAccuracy();

    float value(Tensor *T, Tensor *Y) override;
};

#endif
