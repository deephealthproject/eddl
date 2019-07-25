
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

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "binder.h"


using namespace std;

extern ostream &operator<<(ostream &os, const vector<int> shape);

EDDL_BINDER eddl;

////////////////////////////////////////////////////////
///// EDDL is a wrapper class to ease and define the API
////////////////////////////////////////////////////////

tensor EDDL_BINDER::T(const vector<int> &shape) {
    return new LTensor(shape, DEV_CPU);
}

tensor EDDL_BINDER::T(const vector<int> &shape,float *ptr) {
    return new LTensor(s,ptr,DEV_CPU);
}

tensor EDDL_BINDER::T(string fname) {
    return new LTensor(fname);
}

float * EDDL_BINDER::T_getptr(tensor T)
{
  return T->input->ptr;
}

void EDDL_BINDER::div(tensor t, float v) {
    t->input->div(v);
}

// ---- Operator Layers ----
layer EDDL_BINDER::Abs(layer l) {
    return new LAbs(l, "", DEV_CPU);
}

layer EDDL_BINDER::Diff(layer l1, layer l2) {
    return new LDiff(l1, l2, "", DEV_CPU);
}

layer EDDL_BINDER::Diff(layer l1, float k) {
    return new LDiff(l1, k, "", DEV_CPU);
}

layer EDDL_BINDER::Div(layer l1, layer l2) {
    return new LDiv(l1, l2, "", DEV_CPU);
}

layer EDDL_BINDER::Div(layer l1, float k) {
    return new LDiv(l1, k, "", DEV_CPU);
}

layer EDDL_BINDER::Exp(layer l) {
    return new LExp(l, "", DEV_CPU);
}

layer EDDL_BINDER::Log(layer l) {
    return new LLog(l, "", DEV_CPU);
}

layer EDDL_BINDER::Log2(layer l) {
    return new LLog2(l, "", DEV_CPU);
}

layer EDDL_BINDER::Log10(layer l) {
    return new LLog10(l, "", DEV_CPU);
}

layer EDDL_BINDER::Mult(layer l1, layer l2) {
    return new LMult(l1, l2, "", DEV_CPU);
}

layer EDDL_BINDER::Mult(layer l1, float k) {
    return new LMult(l1, k, "", DEV_CPU);
}

layer EDDL_BINDER::Pow(layer l1, layer l2) {
    return new LPow(l1, l2, "", DEV_CPU);
}

layer EDDL_BINDER::Pow(layer l1, float k) {
    return new LPow(l1, k, "", DEV_CPU);
}

layer EDDL_BINDER::Sqrt(layer l) {
    return new LSqrt(l, "", DEV_CPU);
}

layer EDDL_BINDER::Sum(layer l1, layer l2) {
    return new LSum(l1, l2, "", DEV_CPU);
}

layer EDDL_BINDER::Sum(layer l1, float k) {
    return new LSum(l1, k, "", DEV_CPU);
}


// ---- Reduction Layers ----
layer EDDL_BINDER::ReduceMean(layer l) {
    return EDDL_BINDER::ReduceMean(l, {0}, false);
}

layer EDDL_BINDER::ReduceMean(layer l, const vector<int> axis) {
    return EDDL_BINDER::ReduceMean(l, axis, false);
}

layer EDDL_BINDER::ReduceMean(layer l, bool keepdims) {
    return EDDL_BINDER::ReduceMean(l, {0}, keepdims);
}

layer EDDL_BINDER::ReduceMean(layer l, const vector<int> axis, bool keepdims) {
    return new LRMean(l, axis, keepdims, "", DEV_CPU);
}

layer EDDL_BINDER::ReduceVar(layer l) {
    return EDDL_BINDER::ReduceVar(l, {0});
}

layer EDDL_BINDER::ReduceVar(layer l, const vector<int> axis) {
    return EDDL_BINDER::ReduceVar(l, axis, false);
}

layer EDDL_BINDER::ReduceVar(layer l, bool keepdims) {
    return EDDL_BINDER::ReduceVar(l, {0}, keepdims);
}

layer EDDL_BINDER::ReduceVar(layer l, const vector<int> axis, bool keepdims) {
    return new LRVar(l, axis, keepdims, "", DEV_CPU);
}

layer EDDL_BINDER::ReduceSum(layer l) {
    return EDDL_BINDER::ReduceSum(l, {0});
}

layer EDDL_BINDER::ReduceSum(layer l, vector<int> axis) {
    return EDDL_BINDER::ReduceSum(l, axis, false);
}

layer EDDL_BINDER::ReduceSum(layer l, bool keepdims) {
    return EDDL_BINDER::ReduceSum(l, {0}, keepdims);
}

layer EDDL_BINDER::ReduceSum(layer l, const vector<int> axis, bool keepdims) {
    return new LRSum(l, axis, keepdims, "", DEV_CPU);
}

layer EDDL_BINDER::ReduceMax(layer l) {
    return EDDL_BINDER::ReduceMax(l, {0});
}

layer EDDL_BINDER::ReduceMax(layer l, vector<int> axis) {
    return EDDL_BINDER::ReduceMax(l, axis, false);
}

layer EDDL_BINDER::ReduceMax(layer l, bool keepdims) {
    return EDDL_BINDER::ReduceMax(l, {0}, keepdims);
}

layer EDDL_BINDER::ReduceMax(layer l, const vector<int> axis, bool keepdims) {
    return new LRMax(l, axis, keepdims, "", DEV_CPU);
}

layer EDDL_BINDER::ReduceMin(layer l) {
    return EDDL_BINDER::ReduceMin(l, {0});
}

layer EDDL_BINDER::ReduceMin(layer l, vector<int> axis) {
    return EDDL_BINDER::ReduceMin(l, axis, false);
}

layer EDDL_BINDER::ReduceMin(layer l, bool keepdims) {
    return EDDL_BINDER::ReduceMin(l, {0}, keepdims);
}

layer EDDL_BINDER::ReduceMin(layer l, const vector<int> axis, bool keepdims) {
    return new LRMin(l, axis, keepdims, "", DEV_CPU);
}

// ---- Generator Layers ----
layer EDDL_BINDER::GaussGenerator(float mean, float stdev, vector<int> size) {
    return new LGauss(mean, stdev, size, "", DEV_CPU);
}

layer EDDL_BINDER::UniformGenerator(float low, float high, vector<int> size) {
    return new LUniform(low, high, size, "", DEV_CPU);
}

//////////////////////////////////////////////////////


layer EDDL_BINDER::Activation(layer parent, string activation, string name) {
    return new LActivation(parent, activation, name, DEV_CPU);
}
//////////////////////////////////////////////////////


layer EDDL_BINDER::BatchNormalization(layer parent, float momentum, float epsilon, bool affine, string name){
    return new LBatchNorm(parent, momentum, epsilon, affine, name, DEV_CPU);
}

//////////////////////////////////////////////////////



layer EDDL_BINDER::Conv(layer parent, int filters, const vector<int> &kernel_size,
                 const vector<int> &strides, string padding, int groups, const vector<int> &dilation_rate,
                 bool use_bias, string name) {
    return new LConv(parent, filters, kernel_size, strides, padding, groups, dilation_rate, use_bias, name, DEV_CPU);
}
//////////////////////////////////////////////////////


layer EDDL_BINDER::ConvT(layer parent, int filters, const vector<int> &kernel_size,
                  const vector<int> &output_padding, string padding, const vector<int> &dilation_rate,
                  const vector<int> &strides, bool use_bias, string name){
    return new LConvT(parent, filters, kernel_size, output_padding, padding, dilation_rate, strides, use_bias, name, DEV_CPU);
}
/////////////////////////////////////////////////////////


layer EDDL_BINDER::Dense(layer parent, int ndim, bool use_bias, string name){
    return new LDense(parent, ndim, use_bias, name, DEV_CPU);
}
//////////////////////////////////////////////////////


layer EDDL_BINDER::Dropout(layer parent, float rate, string name) {
    return new LDropout(parent, rate, name, DEV_CPU);
}

//////////////////////////////////////////////////////
layer EDDL_BINDER::Embedding(int input_dim, int output_dim, string name){
    return new LEmbedding(input_dim, output_dim, name, DEV_CPU);
}

//////////////////////////////////////////////////////
layer EDDL_BINDER::GaussianNoise(layer parent, float stdev, string name){
    return new LGaussianNoise(parent, stdev, name, DEV_CPU);
}

//////////////////////////////////////////////////////


layer EDDL_BINDER::Input(const vector<int> &shape, string name) {
    tshape s=vector<int>(shape.begin(), shape.end());
    s.insert(s.begin(), 1);

    return new LInput(new Tensor(s), name, DEV_CPU);
}

//////////////////////////////////////////////////////

layer EDDL_BINDER::UpSampling(layer parent, const vector<int> &size, string interpolation, string name){
    return new LUpSampling(parent, size, interpolation, name, DEV_CPU);
}

//////////////////////////////////////////////////////
layer EDDL_BINDER::AveragePool(layer parent, const vector<int> &pool_size) {
    return EDDL_BINDER::AveragePool(parent, pool_size, pool_size);
}

layer EDDL_BINDER::AveragePool(layer parent, const vector<int> &pool_size, const vector<int> &strides, string padding, string name){
    //TODO: Implement
    return new LAveragePool(parent, pool_size, strides, padding, name, DEV_CPU);
}

//////////////////////////////////////////////////////

layer EDDL_BINDER::GlobalMaxPool(layer parent, string name){
    //TODO: Implement
    //return new LGlobalMaxPool(parent, PoolDescriptor({1,1}, {1,1}), name, DEV_CPU);
    return nullptr;
}

//////////////////////////////////////////////////////

layer EDDL_BINDER::GlobalAveragePool(layer parent, string name){
    //TODO: Implement
    //return new LGlobalAveragePool(parent,  PoolDescriptor({1,1}, {1,1}, "none"), name, DEV_CPU);
    return nullptr;
}


//////////////////////////////////////////////////////

layer EDDL_BINDER::MaxPool(layer parent, const vector<int> &pool_size, string padding, string name){
    return new LMaxPool(parent, pool_size, pool_size, padding, name, DEV_CPU);
}

layer EDDL_BINDER::MaxPool(layer parent, const vector<int> &pool_size, const vector<int> &strides, string padding, string name){
    return new LMaxPool(parent, pool_size, strides, padding, name, DEV_CPU);
}

//////////////////////////////////////////////////////

layer EDDL_BINDER::RNN(layer parent, int units, int num_layers, bool use_bias, float dropout, bool bidirectional, string name){
    return new LRNN(parent, units, num_layers, use_bias, dropout, bidirectional, name, DEV_CPU);
}

//////////////////////////////////////////////////////

layer EDDL_BINDER::LSTM(layer parent, int units, int num_layers, bool use_bias, float dropout, bool bidirectional, string name){
    return new LLSTM(parent, units, num_layers, use_bias, dropout, bidirectional, name, DEV_CPU);
}
//////////////////////////////////////////////////////

layer EDDL_BINDER::Reshape(layer parent, const vector<int> &shape, string name) {
    tshape s=vector<int>(shape.begin(), shape.end());
    s.insert(s.begin(), 1);
    return new LReshape(parent, s, name, DEV_CPU);
}
/////////////////////////////////////////////////////////

layer EDDL_BINDER::Transpose(layer parent, const vector<int> &dims, string name){
    return new LTranspose(parent, dims, name, DEV_CPU);
}
/////////////////////////////////////////////////////////


loss EDDL_BINDER::LossFunc(string type){
    if(type == "mse" || type == "mean_squared_error"){
        return new LMeanSquaredError();
    } else if(type == "cross_entropy"){
        return new LCrossEntropy();
    } else if (type == "soft_cross_entropy"){
        return new LSoftCrossEntropy();
    }
    return nullptr;
}
/////////////////////////////////////////////////////////


metric EDDL_BINDER::MetricFunc(string type){
    if(type == "mse" || type == "mean_squared_error"){
        return new MMeanSquaredError();
    } else if(type == "categorical_accuracy" || type == "accuracy"){
        return new MCategoricalAccuracy();
    }
    return nullptr;
}
/////////////////////////////////////////////////////////



layer EDDL_BINDER::Add(const vector<layer> &layers, string name) {
    return new LAdd(layers, name, DEV_CPU);
}

////////////////////////////////////////////////////////

layer EDDL_BINDER::Average(const vector<layer> &layers, string name){
    //TODO: Implement
    return new LAverage(layers, name, DEV_CPU);
}

/////////////////////////////////////////////////////////

layer EDDL_BINDER::Subtract(const vector<layer> &layers, string name) {
    return new LSubtract(layers, name, DEV_CPU);
}

////////////////////////////////////////////////////////

layer EDDL_BINDER::Concat(const vector<layer> &layers, string name) {
    return new LConcat(layers, name, DEV_CPU);
}

////////////////////////////////////////////////////////

layer EDDL_BINDER::MatMul(const vector<layer> &layers, string name){
    return new LMatMul(layers, name, DEV_CPU);
}

////////////////////////////////////////////////////////

layer EDDL_BINDER::Maximum(const vector<layer> &layers, string name){
    return new LMaximum(layers, name, DEV_CPU);
}

////////////////////////////////////////////////////////

layer EDDL_BINDER::Minimum(const vector<layer> &layers, string name){
    return new LMinimum(layers, name, DEV_CPU);
}


////////////////////////////////////////////////////////

optimizer EDDL_BINDER::adadelta(float lr, float rho, float epsilon, float weight_decay){
    //Todo: Implement
    return new AdaDelta(lr, rho, epsilon, weight_decay);
}
optimizer EDDL_BINDER::adam(float lr, float beta_1, float beta_2, float epsilon, float weight_decay, bool amsgrad){
    //Todo: Implement
    return new Adam(lr, beta_1, beta_2, epsilon, weight_decay, amsgrad);
}
optimizer EDDL_BINDER::adagrad(float lr, float epsilon, float weight_decay){
    //Todo: Implement
    return new Adagrad(lr, epsilon, weight_decay);
}
optimizer EDDL_BINDER::adamax(float lr, float beta_1, float beta_2, float epsilon, float weight_decay){
    //Todo: Implement
    return new Adamax(lr, beta_1, beta_2, epsilon, weight_decay);
}
optimizer EDDL_BINDER::nadam(float lr, float beta_1, float beta_2, float epsilon, float schedule_decay){
    //Todo: Implement
    return new Nadam(lr, beta_1, beta_2, epsilon, schedule_decay);
}
optimizer EDDL_BINDER::rmsprop(float lr, float rho, float epsilon, float weight_decay){
    //Todo: Implement
    return new RMSProp(lr, rho, epsilon, weight_decay);
}

optimizer EDDL_BINDER::sgd(float lr, float momentum, float weight_decay, bool nesterov){
    return new SGD(lr, momentum, weight_decay, nesterov);
}

void EDDL_BINDER::change(optimizer o, vector<float> &params){
    o->change(params);
}

////////////////////////////////////////////////////////
initializer EDDL_BINDER::Constant(float value){
    //Todo: Implement
    return new IConstant(value);
}

initializer EDDL_BINDER::Identity(float gain){
    //Todo: Implement
    return new IIdentity(gain);
}
initializer EDDL_BINDER::GlorotNormal(float seed) {
    //Todo: Implement
    return new IGlorotNormal(seed);
}
initializer EDDL_BINDER::GlorotUniform(float seed){
    //Todo: Implement
    return new IGlorotUniform(seed);
}
initializer EDDL_BINDER::RandomNormal(float mean, float stdev, int seed){
    //Todo: Implement
    return new IRandomNormal(mean, stdev, seed);
}
initializer EDDL_BINDER::RandomUniform(float minval, float maxval, int seed){
    //Todo: Implement
    return new IRandomUniform(minval, maxval, seed);
}
initializer EDDL_BINDER::Orthogonal(float gain, int seed){
    //Todo: Implement
    return new IOrthogonal(gain, seed);
}


/////////////////////////////////////////////////////////
model EDDL_BINDER::Model(vlayer in, vlayer out) {
    return new Net(in, out);
}

///////////
compserv EDDL_BINDER::CS_CPU(int th) {
    return new CompServ(th, {}, {});
}

compserv EDDL_BINDER::CS_GPU(const vector<int> &g) {
    return new CompServ(0, g, {});
}

compserv EDDL_BINDER::CS_FGPA(const vector<int> &f) {
    return new CompServ(0, {}, f);
}


////////////

string EDDL_BINDER::summary(model m) {
    return m->summary();
}

void EDDL_BINDER::save(model m,string fname)
{
  FILE *fe = fopen(fname.c_str(), "wb");
  if (fe == nullptr) {
      fprintf(stderr, "Not able to write to %s \n", fname.c_str());
      exit(1);
  }

  fprintf(stderr, "writting bin file\n");

  m->save(fe);

  fclose(fe);

}

void EDDL_BINDER::load(model m,string fname)
{
  FILE *fe = fopen(fname.c_str(), "rb");
  if (fe == nullptr) {
      fprintf(stderr, "Not able to read from %s \n", fname.c_str());
      exit(1);
  }

  fprintf(stderr, "reading bin file\n");

  m->load(fe);

  fclose(fe);

}
void EDDL_BINDER::plot(model m, string fname) {
    m->plot(fname);
}

void EDDL_BINDER::build(model net, optimizer o, const vector<Loss *> &lo, const vector<Metric *> &me) {
    EDDL_BINDER::build(net, o, lo, me, new CompServ(std::thread::hardware_concurrency(), {}, {}));
}

void EDDL_BINDER::build(model net, optimizer o, const vector<Loss *> &lo, const vector<Metric *> &me, CompServ *cs) {
    net->build(o, lo, me, cs);
}


void EDDL_BINDER::fit(model net, const vector<LTensor *> &in, const vector<LTensor *> &out, int batch, int epochs) {
    vtensor tin;
    for (int i = 0; i < in.size(); i++)
        tin.push_back(in[i]->input);

    vtensor tout;
    for (int i = 0; i < out.size(); i++)
        tout.push_back(out[i]->input);


    net->fit(tin, tout, batch, epochs);
}


void EDDL_BINDER::evaluate(model net, const vector<LTensor *> &in, const vector<LTensor *> &out) {
    vtensor tin;
    for (int i = 0; i < in.size(); i++)
        tin.push_back(in[i]->input);

    vtensor tout;
    for (int i = 0; i < out.size(); i++)
        tout.push_back(out[i]->input);


    net->evaluate(tin, tout);
}

void EDDL_BINDER::predict(model net, const vector<LTensor *> &in, const vector<LTensor *> &out) {
    vtensor tin;
    for (int i = 0; i < in.size(); i++)
        tin.push_back(in[i]->input);

    vtensor tout;
    for (int i = 0; i < out.size(); i++)
        tout.push_back(out[i]->input);


    net->predict(tin, tout);
}


void EDDL_BINDER::set_trainable(layer l){
    //Todo: Implement

}


layer EDDL_BINDER::get_layer(model m, string layer_name){
    //Todo: Implement
    return nullptr;
}


model EDDL_BINDER::load_model(string fname){
    //Todo: Implement
    return nullptr;
}

void EDDL_BINDER::save_model(model m, string fname){
    //Todo: Implement
}


void EDDL_BINDER::set_trainable(model m){
    //Todo: Implement
}

model EDDL_BINDER::zoo_models(string model_name){
    //Todo: Implement
    return nullptr;
}

////

bool exist(string name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    }
    return false;
}

void EDDL_BINDER::download_mnist() {
    // TODO: Too big, we should use the one in the PyEDDL
    // TODO: Need for "to_categorical" method
    string cmd;
    string trX = "trX.bin";
    string trY = "trY.bin";
    string tsX = "tsX.bin";
    string tsY = "tsY.bin";

    if ((!exist(trX)) || (!exist(trY)) || (!exist(tsX)) || (!exist(tsY))) {
        cmd = "wget https://www.dropbox.com/s/khrb3th2z6owd9t/trX.bin";
        int status = system(cmd.c_str());
        if (status < 0) {
            msg("wget must be installed", "eddl.download_mnist");
            exit(1);
        }

        cmd = "wget https://www.dropbox.com/s/m82hmmrg46kcugp/trY.bin";
        status = system(cmd.c_str());
        if (status < 0) {
            msg("wget must be installed", "eddl.download_mnist");
            exit(1);
        }
        cmd = "wget https://www.dropbox.com/s/7psutd4m4wna2d5/tsX.bin";
        status = system(cmd.c_str());
        if (status < 0) {
            msg("wget must be installed", "eddl.download_mnist");
            exit(1);
        }
        cmd = "wget https://www.dropbox.com/s/q0tnbjvaenb4tjs/tsY.bin";
        status = system(cmd.c_str());
        if (status < 0) {
            msg("wget must be installed", "eddl.download_mnist");
            exit(1);
        }

    }
}


model EDDL_BINDER::get_model_mlp(int batch_size){
    // Temp. for debugging
    // network
    layer in=eddl.Input({batch_size, 784});
    layer l=in;

    for(int i=0;i<3;i++)
        l=eddl.Activation(eddl.Dense(l,1024),"relu");

    layer out=eddl.Activation(eddl.Dense(l,10),"softmax");

    // net define input and output layers list
    model net=eddl.Model({in},{out});

    return net;
}

model EDDL_BINDER::get_model_cnn(int batch_size){
    // Temp. for debugging
    // network
    layer in=eddl.Input({batch_size,784});
    layer l=in;

    l=eddl.Reshape(l,{batch_size, 1,28,28});
    l=eddl.MaxPool(eddl.Activation(eddl.Conv(l, 16, {3,3}),"relu"),{2,2});
    l=eddl.MaxPool(eddl.Activation(eddl.Conv(l, 32, {3,3}),"relu"),{2,2});
    l=eddl.MaxPool(eddl.Activation(eddl.Conv(l, 64, {3,3}),"relu"),{2,2});
    l=eddl.MaxPool(eddl.Activation(eddl.Conv(l, 128, {3,3}),"relu"),{2,2});

    /*for(int i=0,k=16;i<3;i++,k=k*2)
      l=ResBlock(l,k,2);
  */
    l=eddl.Reshape(l,{batch_size,-1});

    l=eddl.Activation(eddl.Dense(l,32),"relu");

    layer out=eddl.Activation(eddl.Dense(l,10),"softmax");

    // net define input and output layers list
    model net=eddl.Model({in},{out});
    return net;
}





//////
