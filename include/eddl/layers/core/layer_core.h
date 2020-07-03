/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_CORE_H
#define EDDL_LAYER_CORE_H

#include <string>
#include <cstdio>

#include "eddl/layers/layer.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;

/// Tensor Layer
class LTensor : public LinLayer {
public:

    Tensor *data;

    static int total_layers;

    LTensor(string fname);
    ~LTensor() override;

    LTensor(vector<int> shape, int dev, int mem);

    LTensor(const vector<int> shape, float *fptr,int dev, int mem);

    LTensor *fromCSV(string fname);

    explicit LTensor(Layer *l);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void info() override {}

    void forward() override {}

    void backward() override {}

    string plot(int c) override { return ""; }

    LTensor operator+(LTensor L);


};

/// INPUT Layer
class LInput : public LinLayer {
public:
    static int total_layers;

    LInput(Tensor *in, string name, int dev, int mem);
    ~LInput() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void free_delta() override;

    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// EMBEDDING Layer
class LEmbedding : public LinLayer {
public:
    int dim;
    int vocsize;
    int length;
    bool mask_zeros;
    Tensor *E;
    Tensor *gE;
    vector<int> sind;
    static int total_layers;

    LEmbedding(Layer *parent, int vocsize, int lenght, int dim, bool mask_zeros, string name, int dev, int mem);

    ~LEmbedding();

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// Dense Layer
class LDense : public LinLayer {
public:
    static int total_layers;
    int ndim;
    bool use_bias;  // TODO: Implement
	bool distributed_training;

    LDense(Layer *parent, int ndim, bool use_bias, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    // Params
    Tensor *W;
    Tensor *gW;
	Tensor *acc_gW;
    Tensor *bias;
    Tensor *gbias;
	Tensor *acc_gbias;

    void forward() override;

    void backward() override;

	// Sets the weights to the values of the parameter w
	void update_weights(Tensor* w, Tensor* bias=nullptr) override;

	// Adds the values of gw to the current weights of the layer
	void accumulate_accumulated_gradients(Tensor* gw, Tensor* gbias=nullptr) override;

	// Sets to 0.0 the tensors with the accumulated gradients for W and bias
	void reset_accumulated_gradients() override;

	void apply_accumulated_gradients() override;

    string plot(int c) override;

	static void reset_name_counter();

	void enable_distributed() override;

};

/// Activation Layer
class LActivation : public LinLayer {
public:
    string act;
    static int total_layers;
    vector<float> params;

    LActivation(Layer *parent, string act, vector<float> params, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void save(std::ofstream &ofs, string format) override;
    void load(std::ifstream &ifs, string format) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// Reshape Layer
class LReshape : public LinLayer {
public:
    static int total_layers;
    vector<int> ls;

    // constructors and clones
    LReshape(Layer *parent, vector<int> shape, string name, int dev, int mem);
    ~LReshape() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    // implementation
    void mem_delta() override;
    void free_delta() override;

    void forward() override;

    void backward() override;

    void resize(int batch) override;

    string plot(int c) override;

};

/// Transpose Layer
class LTranspose : public LinLayer {
public:
    static int total_layers;
    vector<int> dims;
    vector<int> rdims;

    // constructors and clones
    LTranspose(Layer *parent, vector<int> dims, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// Drop-out Layer
class LDropout : public LinLayer {
public:
    static int total_layers;
    bool iw; //inference weighting
    
    // constructors and clones
    LDropout(Layer *parent, float df, bool iw, string name, int dev, int mem);
    ~LDropout() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    float df;
    Tensor *mask;

    // implementation
    void forward() override;

    void backward() override;
    void resize(int batch) override;
    string plot(int c) override;

};


/// Select Layer
class LSelect : public LinLayer {
public:
    static int total_layers;
    SelDescriptor *sd;

    LSelect(Layer *l, vector<string> indices, string name, int dev, int mem);

    ~LSelect();

    void forward() override;

    void backward() override;

    void resize(int b) override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};


/// Permute Layer
class LPermute : public LinLayer {
public:
    static int total_layers;

    PermuteDescriptor *sd;

    LPermute(Layer *l, vector<int> dims, string name, int dev, int mem);

    ~LPermute();

    void forward() override;

    void backward() override;

    void resize(int b) override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

#endif //EDDL_LAYER_CORE_H
