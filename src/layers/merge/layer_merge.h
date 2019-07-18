
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

#ifndef EDDLL_LAYER_MERGE_H
#define EDDLL_LAYER_MERGE_H

#include <string>
#include <stdio.h>

#include "../layer.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;


/// Add Layer
class LAdd : public MLayer {
public:
    static int total_layers;


    LAdd(vector<Layer *> in, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// Subtract Layer
class LSubtract : public MLayer {
public:
    static int total_layers;


    LSubtract(vector<Layer *> in, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// MatMul Layer
class LMatMul : public MLayer {
public:
    static int total_layers;


    LMatMul(vector<Layer *> in, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// Average Layer
class LAverage : public MLayer {
public:
    static int total_layers;


    LAverage(vector<Layer *> in, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// Maximum Layer
class LMaximum : public MLayer {
public:
    static int total_layers;


    LMaximum(vector<Layer *> in, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// Maximum Layer
class LMinimum : public MLayer {
public:
    static int total_layers;


    LMinimum(vector<Layer *> in, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// Concat Layer
class LConcat : public MLayer {
public:
    int ndim;
    vector<int> index;
    static int total_layers;

    // constructors and clones
    LConcat(vector<Layer *> in, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    // Params


    // implementation
    void forward() override;

    void backward() override;

    string plot(int c) override;

};

#endif //EDDLL_LAYER_MERGE_H

