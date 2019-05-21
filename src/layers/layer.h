// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019
//           Roberto Paredes Palacios, <rparedes@dsic.upv.es>
//           Jon Ander Gómez, <jon@dsic.upv.es>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef _LAYER_
#define _LAYER_

#include <string>
#include <stdio.h>
#include "../tensor.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;


class Layer {
public:
    string name;
    Tensor *input;
    Tensor *output;
    Tensor *target;
    Tensor *delta;
    Layer *orig;

    vector<Tensor *> params;
    vector<Tensor *> gradients;

    vector<Layer *> parent;
    vector<Layer *> child;

    int mode;
    int dev;
    int lin, lout;
    int delta_bp;

    Layer(string name, int dev);

    void initialize();

    void reset();

    virtual void info();

    void setmode(int m);

    Tensor getWeights();
    Tensor setWeights(Tensor bias);

    Tensor getBias();
    Tensor setBias(Tensor bias);

    //virtual
    virtual string plot(int c) { return ""; }

    virtual void addchild(Layer *l) {}

    virtual void addparent(Layer *l) {}

    virtual void forward() {}

    virtual void backward() {}

    virtual Layer *share(int c, int bs, vector<Layer *> p) { return nullptr; }

    virtual Layer *clone(int c, int bs, vector<Layer *> p, int todev) { return nullptr; }

};


/////////////////////////////////////////
/////////////////////////////////////////
// Operator layer
class OperatorLayer : public Layer {
public:

    OperatorLayer(string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Abs Layer
class LAbs : public OperatorLayer {
public:

    LAbs(Layer *l, string name, int dev);

    void forward() override;

    void backward() override;
};


/// Diff Layer
class LDiff : public OperatorLayer {
public:


    LDiff(Layer *l1, Layer *l2, string name, int dev);
    LDiff(Layer *l, float k, string name, int dev);

    void forward() override;

    void backward() override;
};

/// Div Layer
class LDiv : public OperatorLayer {
public:

    LDiv(Layer *l1, Layer *l2, string name, int dev);
    LDiv(Layer *l, float k, string name, int dev);

    void forward() override;

    void backward() override;
};

/// Exp Layer
class LExp : public OperatorLayer {
public:

    LExp(Layer *l, float k, string name, int dev);

    void forward() override;

    void backward() override;
};

/// Log Layer
class LLog : public OperatorLayer {
public:

    LLog(Layer *l, float k, string name, int dev);

    void forward() override;

    void backward() override;
};

/// Mean Layer
class LMean : public OperatorLayer {
public:

    LMean(Layer *l, string name, int dev);

    void forward() override;

    void backward() override;
};

/// Mult Layer
class LMult : public OperatorLayer {
public:

    LMult(Layer *l1, Layer *l2, string name, int dev);
    LMult(Layer *l, float k, string name, int dev);

    void forward() override;

    void backward() override;
};

/// Pow Layer
class LPow : public OperatorLayer {
public:

    LPow(Layer *l, float k, string name, int dev);

    void forward() override;

    void backward() override;
};

/// Sqrt Layer
class LSqrt : public OperatorLayer {
public:

    LSqrt(Layer *l, float k, string name, int dev);

    void forward() override;

    void backward() override;
};

/// Sum Layer
class LSum : public OperatorLayer {
public:

    LSum(Layer *l1, Layer *l2, string name, int dev);
    LSum(Layer *l, float k, string name, int dev);

    void forward() override;

    void backward() override;
};

/// Var Layer
class LVar : public OperatorLayer {
public:

    LVar(Layer *l, string name, int dev);

    void forward() override;

    void backward() override;
};


/////////////////////////////////////////
/////////////////////////////////////////
// Layers with only one input
class LinLayer : public Layer {
public:

    LinLayer(string name, int dev);

    void addchild(Layer *l) override;

    void addparent(Layer *l) override;
};

/// Tensor Layer
class LTensor : public LinLayer {
public:
    static int total_layers;

    LTensor(string fname);

    LTensor(const initializer_list<int> &init, int dev);

    LTensor(vector<int> shape, int dev);

    explicit LTensor(Layer *l);

    Layer *share(int c, int bs, vector<Layer *> p) override { return nullptr; }

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override { return nullptr; }

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

    LInput(Tensor *in, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

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

    string plot(int c) override;

};

/// Dense Layer
class LDense : public LinLayer {
public:
    int ndim;
    bool use_bias;  // TODO: Implement
    static int total_layers;

    LDense(Layer *parent, int ndim, bool use_bias, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    // Params
    Tensor *W;
    Tensor *gW;
    Tensor *bias;
    Tensor *gbias;

    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// Activation Layer
class LActivation : public LinLayer {
public:
    string act;
    static int total_layers;

    LActivation(Layer *parent, string act, string name, int d);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

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
    LReshape(Layer *parent, const initializer_list<int> &init, string name, int d);

    LReshape(Layer *parent, vector<int> shape, string name, int d);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;


    // implementation
    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// Transpose Layer
class LTranspose : public LinLayer {
public:
    static int total_layers;
    vector<int> dims;

    // constructors and clones
    LTranspose(Layer *parent, const initializer_list<int> &dims, string name, int dev);

//    Layer *share(int c, int bs, vector<Layer *> p) override;
//
//    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
//
//
//    // implementation
//    void forward() override;
//
//    void backward() override;
//
//    string plot(int c) override;

};

/// Conv2D Layer
class LConv : public LinLayer {
public:
    static int total_layers;

    ConvolDescriptor *cd;

    // constructors and clones
    LConv(Layer *parent, const initializer_list<int> &ks, const initializer_list<int> &st, string p, string name, int d);

    LConv(Layer *parent, const initializer_list<int> &ks, const initializer_list<int> &st,
          const initializer_list<int> &p, string name, int d);

    LConv(Layer *parent, int filters, const initializer_list<int> &kernel_size, const initializer_list<int> &strides, string padding,
            int groups, const initializer_list<int> &dilation_rate, bool use_bias, string name, int dev);

    LConv(Layer *parent, const vector<int> &ks, const vector<int> &st, string p, string name, int d);

    LConv(Layer *parent, ConvolDescriptor *cd, string name, int d);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    // Params are in ConvolDescriptor

    // implementation
    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// ConvT2D Layer
class LConvT : public LinLayer {
public:
    static int total_layers;

    // constructors and clones
    LConvT(Layer *parent, int filters, const initializer_list<int> &kernel_size,
        const initializer_list<int> &output_padding, string padding, const initializer_list<int> &dilation_rate,
        const initializer_list<int> &strides, bool use_bias, string name, int dev);

    LConvT(Layer *parent, ConvolDescriptor *cd, string name, int dev);

//    Layer *share(int c, int bs, vector<Layer *> p) override;
//
//    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
//
//    // Params are in ConvolDescriptor
//
//    // implementation
//    void forward() override;
//
//    void backward() override;
//
//    string plot(int c) override;

};

/// UpSampling2D Layer
class LUpSampling : public LinLayer {
public:
    vector<int> size;
    string interpolation;
    static int total_layers;

    // constructors and clones
    LUpSampling(Layer *parent, const initializer_list<int> &size, string interpolation, string name, int dev);

//    Layer *share(int c, int bs, vector<Layer *> p) override;
//
//    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
//
//    // Params are in ConvolDescriptor
//
//    // implementation
//    void forward() override;
//
//    void backward() override;
//
//    string plot(int c) override;

};

/// Pool2D Layer
class LPool : public LinLayer {
public:
    static int total_layers;
    PoolDescriptor *pd;

    // constructors
    LPool(Layer *parent, PoolDescriptor *cd, string name, int d);
};

/// MaxPool2D Layer
class LMaxPool : public LPool {
public:

    // constructors and clones
    LMaxPool(Layer *parent, const initializer_list<int> &ks, const initializer_list<int> &st, string p, string name,
           int d);

    LMaxPool(Layer *parent, const initializer_list<int> &ks, const initializer_list<int> &st,
           const initializer_list<int> &p, string name, int d);

    LMaxPool(Layer *parent, const vector<int> &ks, const vector<int> &st, string p, string name, int d);

    LMaxPool(Layer *parent, PoolDescriptor *cd, string name, int d);

    // Params
    Tensor *indX, *indY;

    // implementation
    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    string plot(int c) override;

};


/// AveragePool2D Layer
class LAveragePool : public LPool {
public:

    // constructors and clones
    LAveragePool(Layer *parent, const initializer_list<int> &pool_size, const initializer_list<int> &strides, string padding, string name, int dev);
    LAveragePool(Layer *parent, PoolDescriptor *D, string name, int dev);

//    // Params
//    Tensor *indX, *indY;
//
//    // implementation
//    void forward() override;
//
//    void backward() override;
//
//    Layer *share(int c, int bs, vector<Layer *> p) override;
//
//    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
//
//    string plot(int c) override;

};

/// GlobalMaxPool2D Layer
class LGlobalMaxPool : public LPool {
public:

    // constructors and clones
    LGlobalMaxPool(Layer *parent, PoolDescriptor *D, string name, int dev);

//    // Params
//    Tensor *indX, *indY;
//
//    // implementation
//    void forward() override;
//
//    void backward() override;
//
//    Layer *share(int c, int bs, vector<Layer *> p) override;
//
//    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
//
//    string plot(int c) override;

};

/// GlobalAveragePool2D Layer
class LGlobalAveragePool : public LPool {
public:

    // constructors and clones
    LGlobalAveragePool(Layer *parent, PoolDescriptor *D, string name, int dev);

//    // Params
//    Tensor *indX, *indY;
//
//    // implementation
//    void forward() override;
//
//    void backward() override;
//
//    Layer *share(int c, int bs, vector<Layer *> p) override;
//
//    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
//
//    string plot(int c) override;

};

/// Drop-out Layer
class LDropout : public LinLayer {
public:
    int ndim;
    static int total_layers;

    // constructors and clones
    LDropout(Layer *parent, float df, string name, int d);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    float df;
    Tensor *mask;

    // implementation
    void forward() override;

    void backward() override;

    string plot(int c) override;

};


/////////////////////////////////////////
/////////////////////////////////////////
// Layers with several inputs (ADD, CAT,...)
class MLayer : public Layer {
public:

    MLayer(string name, int dev);

    void addchild(Layer *l) override;

    void addparent(Layer *l) override;

    //virtual

    string plot(int c) override { return ""; }

    void forward() override {}

    void backward() override {}

    Layer *share(int c, int bs, vector<Layer *> p) override { return nullptr; }

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override { return nullptr; }

};

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
    LConcat(vector<Layer *> in, string name, int d);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    // Params


    // implementation
    void forward() override;

    void backward() override;

    string plot(int c) override;

};


/// BatchNormalization Layer
class LBatchNorm : public LinLayer {
public:
    float momentum;
    float epsilon;
    bool affine;
    static int total_layers;

    LBatchNorm(Layer *parent, float momentum, float epsilon, bool affine, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// GaussianNoise Layer
class LGaussianNoise : public LinLayer {
public:
    float stdev;
    static int total_layers;

    LGaussianNoise(Layer *parent, float stdev, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// RNN Layer
class LRNN : public LinLayer {
public:
    int units;
    int num_layers;
    bool use_bias;
    float dropout;
    bool bidirectional;
    static int total_layers;

    LRNN(Layer *parent, int units, int num_layers, bool use_bias, float dropout, bool bidirectional, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// LSTM Layer
class LLSTM : public LinLayer {
public:
    int units;
    int num_layers;
    bool use_bias;
    float dropout;
    bool bidirectional;
    static int total_layers;

    LLSTM(Layer *parent, int units, int num_layers, bool use_bias, float dropout, bool bidirectional, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};

#endif
