/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef _OPTIM_
#define _OPTIM_

#include <stdio.h>
#include <string>
#include <vector>

#include "../layers/layer.h"

using namespace std;

//shorcuts
//#define SGD new sgd

typedef vector<Layer *> vlayer;
typedef vector<Tensor *> vtensor;

class Optimizer {
public:
    string name;
    vlayer layers;

    Optimizer();

    virtual void setlayers(vlayer l) {}

    virtual void applygrads(int batch) {}

    virtual Optimizer *clone() { return nullptr; }

    virtual void change(vector<float> &p) {}

};

class SGD : public Optimizer {
public:
    float lr;
    float mu;
    float weight_decay;
    bool nesterov;

    vtensor mT;

    explicit SGD(float lr=0.01f, float momentum=0.0f, float weight_decay=0.0f, bool nesterov=false);
    ~SGD();

    Optimizer *clone() override;

    void setlayers(vlayer l) override;

    void applygrads(int batch) override;

    void change(vector<float> &p) override;
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
    ~Adam();
    
    Optimizer *clone();

    void setlayers(vlayer l);

    void applygrads(int batch);

    void change(vector<float> &p);
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

    vtensor mT;

    explicit RMSProp(float lr=0.01f, float rho=0.9f, float epsilon=1e-8f, float weight_decay=0.0f);

//    Optimizer *clone();
//
//    void setlayers(vlayer l);
//
//    void applygrads(int batch);
//
//    void change(const vector<float> &p);
};
#endif

//////////
