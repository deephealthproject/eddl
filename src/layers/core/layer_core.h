/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_CORE_H
#define EDDL_LAYER_CORE_H

#include <string>
#include <stdio.h>

#include "../layer.h"

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

    LTensor(vector<int> shape, int dev);

    LTensor(const vector<int> shape, float *fptr,int dev);

    LTensor *fromCSV(string fname);

    explicit LTensor(Layer *l);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void info() override {}

    void forward() override {}

    void backward() override {}

    void resize(int batch) override;

    string plot(int c) override { return ""; }

    LTensor operator+(LTensor L);


};

/// INPUT Layer
class LInput : public LinLayer {
public:
    static int total_layers;

    LInput(Tensor *in, string name, int dev);
    ~LInput() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    void resize(int batch) override;

    string plot(int c) override;

};

/// EMBEDDING Layer
class LEmbedding : public LinLayer {
public:
    int input_dim;
    int output_dim;
    static int total_layers;

    LEmbedding(int input_dim, int output_dim, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    void resize(int batch) override;

    string plot(int c) override;

};

/// Dense Layer
class LDense : public LinLayer {
public:
    static int total_layers;
    int ndim;
    bool use_bias;  // TODO: Implement
	bool distributed_training;

    LDense(Layer *parent, int ndim, bool use_bias, string name, int dev);

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

    void resize(int batch) override;

    string plot(int c) override;

	static void reset_name_counter();

	void enable_distributed() override;

};

/// Activation Layer
class LActivation : public LinLayer {
public:
    string act;
    static int total_layers;
    float param;

    LActivation(Layer *parent, string act, string name, int dev,float param=0.01);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void save(std::ofstream &ofs, string format) override;
    void load(std::ifstream &ifs, string format) override;

    void forward() override;

    void backward() override;

    void resize(int batch) override;

    string plot(int c) override;

};

/// Reshape Layer
class LReshape : public LinLayer {
public:
    static int total_layers;
    vector<int> ls;

    // constructors and clones
    LReshape(Layer *parent, vector<int> shape, string name, int dev);
    ~LReshape() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    // implementation
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
    LTranspose(Layer *parent, vector<int> dims, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
//
//
//    // implementation
    void forward() override;
//
    void backward() override;
    void resize(int batch) override;

    string plot(int c) override;

};

/// Drop-out Layer
class LDropout : public LinLayer {
public:
    static int total_layers;

    // constructors and clones
    LDropout(Layer *parent, float df, string name, int dev);
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




#endif //EDDL_LAYER_CORE_H
