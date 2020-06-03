/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_OPERATORS_H
#define EDDL_LAYER_OPERATORS_H


#include <string>
#include <cstdio>

#include "eddl/layers/layer.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;


/////////////////////////////////////////
/////////////////////////////////////////
// Operator layer
class OperatorLayer : public Layer {
public:

    int binary;
    float val;

    OperatorLayer(string name, int dev, int mem);

    void addchild(Layer *l) override;

    void addparent(Layer *l) override;

    string plot(int c) override;

};

/// Abs Layer
class LAbs : public OperatorLayer {
public:
    static int total_layers;

    Tensor *mask;

    LAbs(Layer *l, string name, int dev, int mem);

    void forward() override;

    void backward() override;

    void resize(int b) override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Diff Layer
class LDiff : public OperatorLayer {
public:
    static int total_layers;
    int left;
    vector<Tensor *> tin;

    LDiff(Layer *l1, Layer *l2, string name, int dev, int mem);
    LDiff(Layer *l, float k, string name, int dev, int mem);
    LDiff(float k, Layer *l, string name, int dev, int mem);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Div Layer
class LDiv : public OperatorLayer {
public:
    static int total_layers;
    int left;

    LDiv(Layer *l1, Layer *l2, string name, int dev, int mem);
    LDiv(Layer *l, float k, string name, int dev, int mem);
    LDiv(float k, Layer *l,string name, int dev, int mem);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Exp Layer
class LExp : public OperatorLayer {
public:
    static int total_layers;

    LExp(Layer *l, string name, int dev, int mem);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Log Layer
class LLog : public OperatorLayer {
public:
    static int total_layers;

    LLog(Layer *l, string name, int dev, int mem);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Log2 Layer
class LLog2 : public OperatorLayer {
public:
    static int total_layers;

    LLog2(Layer *l, string name, int dev, int mem);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Log10 Layer
class LLog10 : public OperatorLayer {
public:
    static int total_layers;

    LLog10(Layer *l, string name, int dev, int mem);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Mean Layer
/*class LMean : public OperatorLayer {
public:
    static int total_layers;
    tshape os;
    vector<int> axis;
    bool keepdims;

    LMean(Layer *l, vector<int> &axis, bool keepdims, string name, int dev, int mem);
    LMean(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};*/

/// Mult Layer
class LMult : public OperatorLayer {
public:
    static int total_layers;

    LMult(Layer *l1, Layer *l2, string name, int dev, int mem);
    LMult(Layer *l, float k, string name, int dev, int mem);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Pow Layer
class LPow : public OperatorLayer {
public:
    static int total_layers;

    LPow(Layer *l1, Layer *l2, string name, int dev, int mem);
    LPow(Layer *l, float k, string name, int dev, int mem);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Sqrt Layer
class LSqrt : public OperatorLayer {
public:
    static int total_layers;

    LSqrt(Layer *l, string name, int dev, int mem);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Sum Layer
class LSum : public OperatorLayer {
public:
    static int total_layers;

    LSum(Layer *l1, Layer *l2, string name, int dev, int mem);
    LSum(Layer *l, float k, string name, int dev, int mem);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Var Layer
/*class LVar : public OperatorLayer {
public:
    static int total_layers;
    tshape os;
    vector<int> axis;
    bool keepdims;
    Tensor *mean;
    int rsize;
    vector<Layer *> layers;

    LVar(Layer *l, vector<int> &axis, bool keepdims, string name, int dev, int mem);
    LVar(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};*/



#endif //EDDL_LAYER_OPERATORS_H
