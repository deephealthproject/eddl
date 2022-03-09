/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_OPTIM_H
#define EDDL_OPTIM_H

#include <cstdio>
#include <string>
#include <vector>

#include "eddl/layers/layer.h"

using namespace std;

//shorcuts
//#define SGD new sgd

typedef vector<Layer *> vlayer;
typedef vector<Tensor *> vtensor;

class Optimizer {
public:
    string name;
    vlayer layers;
    bool isshared;
    float clip_val;
    Optimizer *orig;

    Optimizer();
    virtual ~Optimizer();

    void set_clip_val(float v);
    void clip();

    virtual void setlayers(vlayer l) {}

    virtual void applygrads(int batch) {}

    virtual Optimizer *clone() { return nullptr; }
    virtual Optimizer *share() { return nullptr; }

    virtual void change(vector<float> p) {}

};

class SGD : public Optimizer {
public:
    float lr;
    float mu;
    float weight_decay;
    bool nesterov;

    vtensor mT;

    explicit SGD(float lr=0.01f, float momentum=0.0f, float weight_decay=0.0f, bool nesterov=false);
    virtual ~SGD();

    Optimizer *clone() override;
    Optimizer *share() override;

    void setlayers(vlayer l) override;

    void applygrads(int batch) override;

    void change(vector<float> p) override;
};

// ---- Adam ----
class Adam: public Optimizer {
public:
    float lr;
    float beta_1;
    float beta_2;
    float epsilon;
    float weight_decay;
    bool amsgrad;
    int t;

    vtensor mT;
    vtensor vT;
    vtensor mCap;
    vtensor vCap;

    explicit Adam(float lr=0.01f, float beta_1=0.9f, float beta_2=0.999f, float epsilon=1e-8f, float weight_decay=0.0f, bool amsgrad=false);
    virtual ~Adam();

    Optimizer *clone() override;
    Optimizer *share() override;

    void setlayers(vlayer l) override;

    void applygrads(int batch) override;

    void change(vector<float> p) override;
};


// ---- AdaDelta ----
class AdaDelta : public Optimizer {
public:
    float lr;
    float rho;
    float epsilon;
    float weight_decay;

    vtensor mT;

    explicit AdaDelta(float lr=0.01f, float rho=0.95f, float epsilon=1e-8f, float weight_decay=0.0f);
    virtual ~AdaDelta();

//    Optimizer *clone();
//
//    void setlayers(vlayer l);
//
//    void applygrads(int batch);
//
//    void change(const vector<float> &p);
};

// ---- Adagrad ----
class Adagrad : public Optimizer {
public:
    float lr;
    float epsilon;
    float weight_decay;

    vtensor mT;

    explicit Adagrad(float lr=0.01f, float epsilon=1e-8f, float weight_decay=0.0f);
    virtual ~Adagrad();

//    Optimizer *clone();
//
//    void setlayers(vlayer l);
//
//    void applygrads(int batch);
//
//    void change(const vector<float> &p);
};

// ---- Adamax ----
class Adamax : public Optimizer {
public:
    float lr;
    float beta_1;
    float beta_2;
    float epsilon;
    float weight_decay;

    vtensor mT;

    explicit Adamax(float lr=0.01f, float beta_1=0.9f, float beta_2=0.999f, float epsilon=1e-8f, float weight_decay=0.0f);
    virtual ~Adamax();

//    Optimizer *clone();
//
//    void setlayers(vlayer l);
//
//    void applygrads(int batch);
//
//    void change(const vector<float> &p);
};

// ---- Nadam ----
class Nadam : public Optimizer {
public:
    float lr;
    float beta_1;
    float beta_2;
    float epsilon;
    float schedule_decay;

    vtensor mT;

    explicit Nadam(float lr=0.01f, float beta_1=0.9f, float beta_2=0.999f, float epsilon=1e-8f, float schedule_decay=0.004f);

    virtual ~Nadam();

//    Optimizer *clone();
//
//    void setlayers(vlayer l);
//
//    void applygrads(int batch);
//
//    void change(const vector<float> &p);
};

// ---- RMSProp ----
class RMSProp : public Optimizer {
public:
    float lr;
    float rho;
    float epsilon;
    float weight_decay;

    vtensor gT;
    vtensor gT1;

    explicit RMSProp(float lr=0.01f, float rho=0.9f, float epsilon=1e-8f, float weight_decay=0.0f);

    virtual ~RMSProp();

    Optimizer *clone() override;
    Optimizer *share() override;

    void setlayers(vlayer l) override;

    void applygrads(int batch) override;

    void change(vector<float> p) override;
};
#endif

//////////
