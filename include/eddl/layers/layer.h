/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_H
#define EDDL_LAYER_H

#include <string>
#include <cstdio>

#include "eddl/initializers/initializer.h"

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/tensor_reduction.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/regularizers/regularizer.h"


#define TRMODE 1
#define TSMODE 0

using namespace std;

class Net;

class Layer {
private:
    int    reference_counter;
    string name_id; // For instance checking. Else, your compiler must have rtti support.

public:
    string name;
    Tensor *input;
    Tensor *output;
    Tensor *target;
    Tensor *delta;
    Layer *orig;
    Layer *sorig;
    Net *net;
    bool trainable;
    int mem_level; // See CS
    bool isrecurrent;
    bool isshared;
    bool iscloned;
    bool isnorm;
    bool isdecoder;
    bool distributed_training;

    vector<Tensor *> params;
    vector<Tensor *> gradients;

    vector<Tensor *> states;
    vector<Tensor *> delta_states;

  	vector<Tensor *> acc_gradients;

    vector<Layer *> parent;
    vector<Layer *> child;
    vector<Layer *> clones;

    Regularizer *reg;
    Initializer *init;

    int mode;
    int dev;
    int lin, lout;
    int delta_bp;
    bool detached;
    bool do_deletes;
    unsigned int verbosity_level = 0;

    Layer(string name, int dev, int mem, const string &name_id="");

    // Destructor
    virtual ~Layer();

    virtual void initialize();

    virtual void info();

    void setmode(int m);
    void check_target();
    void detach(Layer *l);
    vector<int> getShape();

    void clamp(float min,float max);
    void set_detach();

    void set_mem_level(int mem);

    int decrease_and_get_reference_counter();
    void increase_reference_counter();
    string get_name_id();

    //virtual
    virtual void mem_delta_parent();
    virtual void mem_delta();
    virtual void free_delta();

    virtual void copy(Layer *l2);

    virtual void resize(int batch);
    virtual void setTrainable(bool value);

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

    LinLayer(string name, int dev, int mem, const string &name_id="");

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
