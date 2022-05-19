/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_AUXILIAR_H
#define EDDL_LAYER_AUXILIAR_H


#include <string>
#include <cstdio>

#include "eddl/layers/layer.h"
#include "eddl/layers/merge/layer_merge.h"

using namespace std;

/// Shape Layer
class LShape : public LinLayer {
public:
    static int total_layers;
    vector<float> data;
    bool include_batch;

    LShape(Layer *parent, bool include_batch, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Where Layer
class LWhere : public MLayer {
public:
    static int total_layers;
    Tensor* condition;
    Tensor* t_parent1;
    Tensor* t_parent2;

    LWhere(Layer *parent1, Layer *parent2, Layer *condition, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};

/// Equal Layer
class LEqual : public LinLayer {
public:
    static int total_layers;
    Tensor *A;
    Tensor *B;

    LEqual(Layer *parent1, Layer *parent2, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};

/// ConstOfTensor Layer
class LConstOfTensor : public LinLayer {
public:
    static int total_layers;
    Tensor *const_tensor;

    LConstOfTensor(Tensor* const_tensor, string name, int dev, int mem);
    ~LConstOfTensor() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void free_delta() override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};

/// Gather Layer
class LGather : public LinLayer {
public:
    static int total_layers;
    int axis;
    Tensor *indices;
    GatherDescriptor *sd;

    LGather(Layer *parent, int axis, Tensor* indices, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};

/// Expand Layer
class LExpand : public LinLayer {
public:
    static int total_layers;
    int size;
    ExpandDescriptor *sd;

    LExpand(Layer *parent, int size, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};

/// MultiThreshold Layer
class LMultiThreshold : public LinLayer {
public:

      static int total_layers;
      int size;

      Tensor *thresholds;
      float out_bias;
      float out_scale;

      LMultiThreshold(Layer *parent, vector<int> thresholds_shape, string name, int dev, int mem, float out_bias, float out_scale);

      Layer *share(int c, int bs, vector<Layer *> p) override;

      Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

      void resize(int batch) override;

      void forward() override;

      void backward() override;

      string plot(int c) override;
};

/// TopK Layer
class LTopK : public LinLayer {
public:

	static int total_layers;
	int axis;
	int largest;
	int sorted;
	int K;

	LTopK(Layer *parent, vector<int> K_shape, string name, int dev, int mem, int axis, int largest, int sorted, int K);

      Layer *share(int c, int bs, vector<Layer *> p) override;

      Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

      void resize(int batch) override;

      void forward() override;

      void backward() override;

      string plot(int c) override;
};

/// QuantizeLinear Layer
class LQuantizeLinear : public LinLayer {
public:

	static int total_layers;

	LQuantizeLinear(Layer *parent, string name, int dev, int mem);

      Layer *share(int c, int bs, vector<Layer *> p) override;

      Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

      void resize(int batch) override;

      void forward() override;

      void backward() override;

      string plot(int c) override;
};

/// DequantizeLinear Layer
class LDequantizeLinear : public LinLayer {
public:

	static int total_layers;
    vector<int> shape;

	LDequantizeLinear(Layer *parent, vector<int> shape, string name, int dev, int mem);

      Layer *share(int c, int bs, vector<Layer *> p) override;

      Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

      void resize(int batch) override;

      void forward() override;

      void backward() override;

      string plot(int c) override;
};

#endif //EDDL_LAYER_AUXILIAR_H
