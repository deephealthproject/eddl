/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_H
#define EDDL_LAYER_H

#include <string>
#include <cstdio>

#include "initializers/initializer.h"

#include "tensor/tensor.h"
#include "tensor/tensor_reduction.h"
#include "tensor/nn/tensor_nn.h"
#include "regularizers/regularizer.h"


#define TRMODE 1
#define TSMODE 0

using namespace std;

class Net;

class Layer {
public:
    string name;
    Tensor *input;
    Tensor *output;
    Tensor *target;
    Tensor *delta;
    Layer *orig;
    Net *net;
    bool trainable;
    int mem_level; // See CS

    vector<Tensor *> params;
    vector<Tensor *> gradients;
	vector<Tensor *> acc_gradients;

    vector<Layer *> parent;
    vector<Layer *> child;

    Regularizer *reg;
    Initializer *init;

    int mode;
    int dev;
    int lin, lout;
    int delta_bp;
    bool detached;
    unsigned int verbosity_level = 0;

    Layer(string name, int dev, int mem);
    // Destructor
    virtual ~Layer();

    virtual void initialize();

    virtual void info();

    void setmode(int m);
    void check_target();
    void detach(Layer *l);
    vector<int> getShape();

    Tensor* getWeights();
    Tensor* setWeights(Tensor bias);

    Tensor* getBias();
    Tensor* setBias(Tensor bias);

    void clamp(float min,float max);
    void set_detach();

    void set_mem_level(int mem);

    virtual void mem_delta_parent();
    virtual void mem_delta();
    virtual void free_delta();


    //virtual
    virtual void copy(Layer *l2);

    virtual void resize(int batch);
    virtual void set_trainable(bool value);

    virtual void save(std::ofstream &ofs, string format="");
    virtual void load(std::ifstream &ifs, string format="");

    virtual void reset();
    virtual int get_trainable_params_count();
    virtual void zeroGrads();
    virtual string plot(int c) { return ""; }

    virtual void addchild(Layer *l) {}

    virtual void addparent(Layer *l) {}

    virtual void forward() {}

    virtual void backward() {}

	virtual void update_weights(Tensor* w, Tensor* bias) {}

	virtual void accumulate_accumulated_gradients(Tensor* gw, Tensor* gbias) {}

	virtual void reset_accumulated_gradients() {}

	virtual void apply_accumulated_gradients() {}

    virtual Layer *share(int c, int bs, vector<Layer *> p) { return nullptr; }

    virtual Layer *clone(int c, int bs, vector<Layer *> p, int todev) { return nullptr; }

	virtual void enable_distributed() {}

};



/////////////////////////////////////////
/////////////////////////////////////////
// Layers with only one input
class LinLayer : public Layer {
public:

    LinLayer(string name, int dev, int mem);

    void addchild(Layer *l) override;

    void addparent(Layer *l) override;

    //virtual

    string plot(int c) override { return ""; }

    void forward() override {}

    void backward() override {}

	void update_weights(Tensor* w, Tensor* bias) override {}

	void accumulate_accumulated_gradients(Tensor* gw, Tensor* gbias) override {}

	void reset_accumulated_gradients() override {}

	void apply_accumulated_gradients() override {}

    Layer *share(int c, int bs, vector<Layer *> p) override { return nullptr; }

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override { return nullptr; }

	void enable_distributed() override {};

};

/////////////////////////////////////////
/////////////////////////////////////////
// Layers with several inputs (ADD, CAT,...)
class MLayer : public Layer {
public:

    MLayer(string name, int dev, int mem);

    void addchild(Layer *l) override;

    void addparent(Layer *l) override;

    //virtual

    string plot(int c) override { return ""; }

    void forward() override {}

    void backward() override {}

    Layer *share(int c, int bs, vector<Layer *> p) override { return nullptr; }

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override { return nullptr; }

};

#endif //EDDL_LAYER_H
