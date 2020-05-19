/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "eddl/apis/eddl.h"


using namespace std;

////////////////////////////////////////////////////////
///// EDDL is a wrapper class to ease and define the API
////////////////////////////////////////////////////////


namespace eddl {

    ///////////////////////////////////////
    //  MODEL METHODS
    ///////////////////////////////////////

    // Creation
    model Model(vlayer in, vlayer out){
        return new Net(in, out);
    }

    model Model(vector<Net*> vnets) {
      return new Net(vnets);
    }

    void build(model net, optimizer o, CompServ *cs, bool init_weights){
        // Assign default computing service
        if (cs== nullptr){
            cs = new CompServ(std::thread::hardware_concurrency(), {}, {});
        }
        if (o== nullptr){
            o = new SGD(0.001,0.9);
        }

        net->build(o, {}, {}, cs, init_weights);
    }

    void build(model net, optimizer o, const vector<string> &lo, const vector<string> &me, CompServ *cs, bool init_weights){
        vector<Loss *> l;
        vector<Metric *> m;

        // Replace string by functions
        for (const auto &li : lo){
            l.push_back(getLoss(li));
        }
        for (const auto &mi : me){
            m.push_back(getMetric(mi));
        }

        // Assign default computing service
        if (cs== nullptr){
            cs = new CompServ(std::thread::hardware_concurrency(), {}, {});
        }


        net->build(o, l, m, cs, init_weights);
    }

    // Computing services

    // GPU
    void toGPU(model net)
    {
      net->toGPU({1},1,0);
    }
    void toGPU(model net, vector<int> g)
    {
      net->toGPU(g,1,0);
    }
    void toGPU(model net, vector<int> g,int lsb)
    {
      net->toGPU(g,lsb,0);
    }
    void toGPU(model net, vector<int> g,string mem)
    {
      if (mem=="low_mem") net->toGPU(g,1,2);
      else if (mem=="mid_mem") net->toGPU(g,1,1);
      else if (mem=="full_mem") net->toGPU(g,1,0);
      else msg("Error mem param","toGPU");
    }
    void toGPU(model net, string mem)
    {
      if (mem=="low_mem") net->toGPU({1},1,2);
      else if (mem=="mid_mem") net->toGPU({1},1,1);
      else if (mem=="full_mem") net->toGPU({1},1,0);
      else msg("Error mem param","toGPU");
    }
    void toGPU(model net, vector<int> g,int lsb,string mem)
    {
      if (mem=="low_mem") net->toGPU(g,lsb,2);
      else if (mem=="mid_mem") net->toGPU(g,lsb,1);
      else if (mem=="full_mem") net->toGPU(g,lsb,0);
      else msg("Error mem param","toGPU");
    }

    // CPU
    void toCPU(model net, int t)
    {
        net->toCPU(t);
    }

    compserv CS_CPU(){
        return CS_CPU(-1, "full_mem");
    }

    compserv CS_CPU(int th){
        return CS_CPU(th, "full_mem");
    }

    compserv CS_CPU(int th,string mem){
      if (mem=="low_mem") return new CompServ(th, {}, {}, 0, 2);
      else if (mem=="mid_mem") return new CompServ(th, {}, {}, 0, 1);
      else if (mem=="full_mem") return new CompServ(th, {}, {}, 0, 0);
      else msg("Error mem param","CS_CPU"); // Exits
      return nullptr; // To silent warnings
    }

    compserv CS_GPU(const vector<int> g){
        return CS_GPU(g, 1, "full_mem");
    }
    compserv CS_GPU(const vector<int> g, string mem){
        return CS_GPU(g, 1, mem);
    }
    compserv CS_GPU(const vector<int> g, int lsb){
        return CS_GPU(g, lsb, "full_mem");
    }
    compserv CS_GPU(const vector<int> g, int lsb, string mem){
        if (mem=="low_mem") return new CompServ(0, g, {}, lsb, 2);
        else if (mem=="mid_mem") return new CompServ(0, g, {}, lsb, 1);
        else if (mem=="full_mem") return new CompServ(0, g, {}, lsb, 0);
        else msg("Error mem param","CS_GPU"); // Exits
        return nullptr; // To silent warnings
    }


    compserv CS_FGPA(const vector<int> &f,int lsb){
        return new CompServ(0, {}, f,lsb);
    }

    compserv CS_COMPSS(string filename){
        return new CompServ(filename);
    }

    // Info and logs
    void setlogfile(model net,string fname)
    {
        net->setlogfile(fname);
    }
    void summary(model m){
        cout<<m->summary()<<"\n";
    }
    void plot(model m, string fname,string mode){
        m->plot(fname,mode);
    }

    // Serialization
    void load(model m, const string&  fname, string format){
        m->load(fname,format);
    }

    void save(model m, const string&  fname, string format){
        m->save(fname,format);
    }

    // Optimizer
    void setlr(model net,vector<float>p)
    {
        net->setlr(p);
    }
    optimizer adadelta(float lr, float rho, float epsilon, float weight_decay){
        //Todo: Implement
        return new AdaDelta(lr, rho, epsilon, weight_decay);
    }

    optimizer adam(float lr, float beta_1, float beta_2, float epsilon, float weight_decay, bool amsgrad){
        //Todo: Implement
        return new Adam(lr, beta_1, beta_2, epsilon, weight_decay, amsgrad);
    }

    optimizer adagrad(float lr, float epsilon, float weight_decay){
        //Todo: Implement
        return new Adagrad(lr, epsilon, weight_decay);
    }

    optimizer adamax(float lr, float beta_1, float beta_2, float epsilon, float weight_decay){
        //Todo: Implement
        return new Adamax(lr, beta_1, beta_2, epsilon, weight_decay);
    }

    optimizer nadam(float lr, float beta_1, float beta_2, float epsilon, float schedule_decay){
        //Todo: Implement
        return new Nadam(lr, beta_1, beta_2, epsilon, schedule_decay);
    }

    optimizer rmsprop(float lr, float rho, float epsilon, float weight_decay){
        //Todo: Implement
        return new RMSProp(lr, rho, epsilon, weight_decay);
    }

    optimizer sgd(float lr, float momentum, float weight_decay, bool nesterov){
        return new SGD(lr, momentum, weight_decay, nesterov);
    }

    // Training and Evaluation
    // Coarse methods
    void fit(model net, const vector<Tensor *> &in, const vector<Tensor *> &out, int batch, int epochs){
        net->fit(in, out, batch, epochs);
    }
    void evaluate(model net, const vector<Tensor *> &in, const vector<Tensor *> &out){
        net->evaluate(in, out);
    }
    vector<Tensor *>  predict(model m, const vector<Tensor *> &in)
    {
      return m->predict(in);
    }

    // Finer methods
    vector<int> random_indices(int batch_size, int num_samples){
        vector<int> sind;
        for (int k = 0; k < batch_size; k++) sind.push_back(rand() % num_samples);
        return sind;
    }
    void train_batch(model net, vector<Tensor *> in, vector<Tensor *> out, vector<int> indices){
        net->tr_batches++;
        net->train_batch(in, out, indices);
    }
    void eval_batch(model net, vector<Tensor *> in, vector<Tensor *> out, vector<int> indices){
        net->train_batch(in, out, indices,1);
    }

    void next_batch(vector<Tensor *> in,vector<Tensor *> out)
    {
        int i,n;
        int batch_size;

        batch_size=out[0]->shape[0];
        n=in[0]->shape[0];
        vector<int> sind(batch_size);
        for (i = 0; i < batch_size; i++) sind[i] = rand() % n;

        for (i = 0; i<in.size();i++)
            Tensor::select(in[i], out[i], sind, 0, batch_size);
    }

    void train_batch(model net, vector<Tensor *> in, vector<Tensor *> out){
        net->tr_batches++;
        vector<int> indices;

        for(int i=0;i<in[0]->shape[0];i++) indices.push_back(i);

        net->train_batch(in, out, indices);
    }
    void eval_batch(model net, vector<Tensor *> in, vector<Tensor *> out){
        vector<int> indices;
        for(int i=0;i<in[0]->shape[0];i++) indices.push_back(i);

        net->train_batch(in, out, indices,1);
    }


    // Finest methods
    void set_mode(model net, int mode){
        net->setmode(mode);
    }
    vlayer forward(model net,vector<Layer*> in)
    {
        net->reset();
        net->forward(in);

        return getOut(net);
    }
    vlayer forward(model net,vector<Tensor*> in)
    {
        net->reset();
        net->forward(in);

        return getOut(net);
    }

    vlayer forward(model net,int b)
    {
        net->resize(b);
        net->reset();
        net->forward();

        return getOut(net);

    }
    vlayer forward(model net)
    {
        net->reset();
        net->forward();

        return getOut(net);
    }
    void zeroGrads(model net)
    {
        net->reset_grads();
    }
    void backward(model net,vector<Tensor *> target)
    {
        net->backward(target);
    }
    void backward(model net)
    {
        net->backward({});
    }
    void backward(loss l)
    {
        l->graph->backward();
    }
    void optimize(loss l)
    {
        l->graph->backward();
    }
    void optimize(vector <loss> vl)
    {
      for(auto &l : vl)
        l->graph->backward();
    }

    void update(model net)
    {
        net->update();
    }
    void reset_loss(model m)
    {
        m->reset_loss();
    }
    void print_loss(model m, int batch){
        m->print_loss(batch);
    }

    // model constraints
    void clamp(model m,float min,float max)
    {
        m->clamp(min,max);
    }


    // loss and metrics methods
    float compute_loss(loss L)
    {
        return L->compute();
    }

    float compute_metric(loss L)
    {
        return L->compute();
    }

    Loss* getLoss(string type){
        if (type == "mse" || type == "mean_squared_error"){
            return new LMeanSquaredError();
        } else if (type == "cross_entropy"){
            return new LCrossEntropy();
        } else if (type == "soft_cross_entropy"){
            return new LSoftCrossEntropy();
        }
        else if (type == "dice"){
            return new LDice();
        }
        return nullptr;
    }

    loss newloss(const std::function<Layer*(vector<Layer*>)>& f, vector<Layer*> in, string name)
    {
        return new NetLoss(f,in,name);
    }
    loss newloss(const std::function<Layer*(Layer*)>& f, Layer *in, string name)
    {
        return new NetLoss(f,in,name);
    }
    Metric* getMetric(string type){
        if (type == "mse" || type == "mean_squared_error"){
            return new MMeanSquaredError();
        } else if (type == "categorical_accuracy" || type == "accuracy"){
            return new MCategoricalAccuracy();
        }
        else if (type=="mean_absolute_error"){
            return new MMeanAbsoluteError();
        }
        else if (type=="mean_relative_error"){
            return new MMeanRelativeError();
        }
        else if (type=="dice") {
            return new MDice();
        }
        else if (type=="none") {
            return new Metric("none");
        }
        else {
            throw std::invalid_argument("unsupported metric: " + type);
        }
        return nullptr;
    }
    loss newmetric(const std::function<Layer*(vector<Layer*>)>& f, vector<Layer*> in, string name)
    {
        return new NetLoss(f,in,name);
    }
    loss newmetric(const std::function<Layer*(Layer*)>& f, Layer *in, string name)
    {
        return new NetLoss(f,in,name);
    }


    // graph connections
    layer detach(layer l){
        l->set_detach();
        return l;
    }

    vlayer detach(vlayer l){
        for(int i=0;i<l.size();i++){
            l[i]->set_detach();
        }
        return l;
    }


    ///////////////////////////////////////
    //  LAYERS
    ///////////////////////////////////////

    // Core Layers// ---- CORE LAYERS ----
    layer Activation(layer parent, string activation, vector<float> params, string name){
        return new LActivation(parent, activation, params, name, DEV_CPU, 0);
    }

    layer Softmax(layer parent, string name){
        vector<float> params = {};
        return new LActivation(parent,"softmax", params, name, DEV_CPU, 0);
    }

    layer Sigmoid(layer parent, string name){
        vector<float> params = {};
        return new LActivation(parent,"sigmoid", params, name, DEV_CPU, 0);
    }

    layer HardSigmoid(layer parent, string name){
        vector<float> params = {};
        return new LActivation(parent,"hard_sigmoid", params, name, DEV_CPU, 0);
    }

    layer ReLu(layer parent, string name){
        vector<float> params = {};
        return new LActivation(parent,"relu", params, name, DEV_CPU, 0);
    }

    layer ThresholdedReLu(layer parent, float alpha, string name){
        vector<float> params = {alpha};
        return new LActivation(parent,"thresholded_relu", params, name, DEV_CPU, 0);
    }

    layer LeakyReLu(layer parent, float alpha, string name){
        vector<float> params = {alpha};
        return new LActivation(parent, "leaky_relu", params, name, DEV_CPU, 0);
    }

    layer Elu(layer parent, float alpha, string name){
        vector<float> params = {alpha};
        return new LActivation(parent, "elu", params, name, DEV_CPU, 0);
    }

    layer Selu(layer parent, string name){
        float alpha = 1.6732632423543772848170429916717f;
        float scale = 1.0507009873554804934193349852946f;

        vector<float> params = {alpha, scale};
        return new LActivation(parent, "selu", params, name, DEV_CPU, 0);
    }

    layer Exponential(layer parent, string name){
        vector<float> params = {};
        return new LActivation(parent, "exp", params, name, DEV_CPU, 0);
    }

    layer Softplus(layer parent, string name){
        vector<float> params = {};
        return new LActivation(parent, "softplus", params, name, DEV_CPU, 0);
    }

    layer Softsign(layer parent, string name){
        vector<float> params = {};
        return new LActivation(parent, "softsign", params, name, DEV_CPU, 0);
    }

    layer Tanh(layer parent, string name){
        vector<float> params = {};
        return new LActivation(parent, "tanh", params, name, DEV_CPU, 0);
    }

    layer Linear(layer parent, float alpha, string name){
        vector<float> params = {alpha};
        return new LActivation(parent, "linear",  params, name, DEV_CPU, 0);
    }

    layer Conv(layer parent, int filters, const vector<int> &kernel_size,
               const vector<int> &strides, string padding,  bool use_bias,
               int groups, const vector<int> &dilation_rate,string name){
        return new LConv(parent, filters, kernel_size, strides, padding, groups, dilation_rate, use_bias, name, DEV_CPU, 0);
    }

    layer Conv1D(layer parent, int filters, vector<int> kernel_size,
               vector<int> strides, string padding,  bool use_bias,
               int groups, vector<int> dilation_rate,string name){

        vector<int> shape=parent->output->getShape();
        shape.push_back(1);
        LReshape *l=new LReshape(parent, shape, "", DEV_CPU, 0);

        kernel_size.push_back(1);
        strides.push_back(1);
        LConv *lc=new LConv(l, filters, kernel_size, strides, padding, groups, dilation_rate, use_bias, name, DEV_CPU, 0);

        vector<int> shape2=lc->output->getShape();
        shape2.pop_back();
        return new LReshape(lc,shape2, "", DEV_CPU, 0);

    }





    layer ConvT(layer parent, int filters, const vector<int> &kernel_size,
                const vector<int> &output_padding, string padding, const vector<int> &dilation_rate,
                const vector<int> &strides, bool use_bias, string name){
        return new LConvT(parent, filters, kernel_size, output_padding, padding, dilation_rate, strides, use_bias, name, DEV_CPU, 0);
    }

    layer Dense(layer parent, int ndim, bool use_bias, string name){
        return new LDense(parent, ndim, use_bias, name, DEV_CPU, 0);
    }

  layer Dropout(layer parent, float rate, bool iw, string name){
    return new LDropout(parent, rate, iw, name, DEV_CPU, 0);
    }

    layer Embedding(layer parent, int vocsize, int length, int output_dim,  bool mask_zeros, string name){
        return new LEmbedding(parent, vocsize, length, output_dim, mask_zeros, name, DEV_CPU, 0);
    }

    layer Input(const vector<int> &shape, string name){
        tshape s = vector<int>(shape.begin(), shape.end());
        s.insert(s.begin(), 1);
        return new LInput(new Tensor(s), name, DEV_CPU, 0);
    }

    layer UpSampling(layer parent, const vector<int> &size, string interpolation, string name){
        return new LUpSampling(parent, size, interpolation, name, DEV_CPU, 0);
    }

    layer Reshape(layer parent, const vector<int> &shape, string name){
        tshape s = vector<int>(shape.begin(), shape.end());
        s.insert(s.begin(), 1);
        return new LReshape(parent, s, name, DEV_CPU, 0);
    }

    layer Flatten(layer parent, string name){
        return Reshape(parent, {-1}, name);
    }

    layer Transpose(layer parent, string name){
        vector<int> dims;
        bool ignoreBatch = true;
        int ndims = parent->output->ndim;
        if(ndims<2){
            msg("The parent needs to output a tensor with at least two dimensions", "EDDL::Transpose");
        }

        // Build dimension vector (ignore batch)
        for(int i=(int)ignoreBatch; i < ndims; i++){
            dims.push_back(i-(int)ignoreBatch);
        }
        swap(dims[(ndims-2-(int)ignoreBatch)], dims[(ndims-1-(int)ignoreBatch)]);  // Swap last two indices

        return new LPermute(parent, dims, name, DEV_CPU, 0);
    }


    // Transformation Layers
    layer Shift(layer parent, vector<int> shift, string da_mode, float constant, string name){
        return new LShift(parent, shift, getWrappingMode(da_mode), constant, name, DEV_CPU, 0);
    }
    layer Rotate(layer parent, float angle, vector<int> offset_center, string da_mode, float constant, string name){
        return new LRotate(parent, angle, offset_center, getWrappingMode(da_mode), constant, name, DEV_CPU, 0);
    }

    layer Scale(layer parent, vector<int> new_shape, bool reshape, string da_mode, float constant, string name){
        return new LScale(parent, new_shape, reshape, getWrappingMode(da_mode), constant, name, DEV_CPU, 0);
    }

    layer Flip(layer parent, int axis, string name){
        return new LFlip(parent, axis, name, DEV_CPU, 0);
    }

    layer HorizontalFlip(layer parent, string name){
        return Flip(parent, 0, name);
    }

    layer VerticalFlip(layer parent, string name){
        return Flip(parent, 1, name);
    }


    layer Crop(layer parent, vector<int> from_coords, vector<int> to_coords, bool reshape, float constant, string name){
        return new LCrop(parent, from_coords, to_coords, reshape, constant, name, DEV_CPU, 0);
    }

    layer CenteredCrop(layer parent, vector<int> size, bool reshape, float constant, string name){
        // Compute center
        int center_y = (int)parent->output->shape[2]/2;
        int center_x = (int)parent->output->shape[3]/2;

        // Get coordinates
        int y1 = center_y - size[0]/2;
        int x1 = center_x - size[1]/2;
        int y2 = y1 + size[0] - 1;
        int x2 = x1 + size[1] - 1;
        return new LCrop(parent, {y1, x1}, {y2, x2}, reshape, constant, name, DEV_CPU, 0);
    }

    layer CropScale(layer parent, vector<int> from_coords, vector<int> to_coords, string da_mode, float constant, string name){
        return new LCropScale(parent, from_coords, to_coords, getWrappingMode(da_mode), constant, name, DEV_CPU, 0);
    }

    layer Cutout(layer parent, vector<int> from_coords, vector<int> to_coords, float constant, string name){
        return new LCutout(parent, from_coords, to_coords, constant, name, DEV_CPU, 0);
    }

    // Data augmentation Layers
    layer RandomShift(layer parent, vector<float> factor_x, vector<float> factor_y, string da_mode, float constant, string name){
        return new LShiftRandom(parent, factor_x, factor_y, getWrappingMode(da_mode), constant, name, DEV_CPU, 0);
    }

    layer RandomRotation(layer parent, vector<float> factor, vector<int> offset_center, string da_mode, float constant, string name){
        return new LRotateRandom(parent, factor, offset_center, getWrappingMode(da_mode), constant, name, DEV_CPU, 0);
    }

    layer RandomScale(layer parent, vector<float> factor, string da_mode, float constant, string name){
        return new LScaleRandom(parent, factor, getWrappingMode(da_mode), constant, name, DEV_CPU, 0);
    }

    layer RandomFlip(layer parent, int axis, string name){
        return new LFlipRandom(parent, axis, name, DEV_CPU, 0);
    }

    layer RandomHorizontalFlip(layer parent, string name){
        return RandomFlip(parent, 0, name);
    }

    layer RandomVerticalFlip(layer parent, string name){
        return RandomFlip(parent, 1, name);
    }

    layer RandomCrop(layer parent, vector<int> new_shape, string name){
        return new LCropRandom(parent, new_shape, name, DEV_CPU, 0);
    }

    layer RandomCropScale(layer parent, vector<float> factor, string da_mode, string name){
        return new LCropScaleRandom(parent, factor, getWrappingMode(da_mode), name, DEV_CPU, 0);
    }

    layer RandomCutout(layer parent, vector<float> factor_x, vector<float> factor_y, float constant, string name){
        return new LCutoutRandom(parent, factor_x, factor_y, constant, name, DEV_CPU, 0);
    }

    // Merge Layers
    layer Add(const vector<layer> &layers, string name){
        return new LAdd(layers, name, DEV_CPU, 0);
    }

    layer Average(const vector<layer> &layers, string name){
        //TODO: Implement
        return new LAverage(layers, name, DEV_CPU, 0);
    }

    layer Concat(const vector<layer> &layers, unsigned int axis, string name){
        return new LConcat(layers, axis, name, DEV_CPU, 0);
    }

    layer MatMul(const vector<layer> &layers, string name){
        return new LMatMul(layers, name, DEV_CPU, 0);
    }

    layer Maximum(const vector<layer> &layers, string name){
        return new LMaximum(layers, name, DEV_CPU, 0);
    }

    layer Minimum(const vector<layer> &layers, string name){
        return new LMinimum(layers, name, DEV_CPU, 0);
    }

    layer Subtract(const vector<layer> &layers, string name){
        return new LSubtract(layers, name, DEV_CPU, 0);
    }


    // Noise Layers
    layer GaussianNoise(layer parent, float stdev, string name){
        return new LGaussianNoise(parent, stdev, name, DEV_CPU, 0);
    }


    // Normalization
    layer BatchNormalization(layer parent, float momentum, float epsilon, bool affine, string name){
        return new LBatchNorm(parent, momentum, epsilon, affine, name, DEV_CPU, 0);
    }
    layer BatchNormalization(layer parent, bool affine, float momentum, float epsilon,  string name){
        return new LBatchNorm(parent, momentum, epsilon, affine, name, DEV_CPU, 0);
    }

    layer LayerNormalization(layer parent, float epsilon, bool affine, string name){
        return new LLayerNorm(parent,  epsilon, affine, name, DEV_CPU, 0);
    }
    layer LayerNormalization(layer parent, bool affine,float epsilon,  string name)
    {
        return new LLayerNorm(parent,  epsilon, affine, name, DEV_CPU, 0);
    }


    layer GroupNormalization(layer parent, int groups,float epsilon, bool affine, string name){
        return new LGroupNorm(parent, groups,epsilon,affine,name, DEV_CPU, 0);
    }


    layer Norm(layer parent, float epsilon, string name){
        return new LNorm(parent, epsilon, name, DEV_CPU, 0);
    }

    layer NormMax(layer parent, float epsilon, string name)
    {
        return new LNormMax(parent, epsilon, name, DEV_CPU, 0);
    }

    layer NormMinMax(layer parent, float epsilon, string name)
    {
        return new LNormMinMax(parent, epsilon, name, DEV_CPU, 0);
    }




    //  Operator Layers
    layer Abs(layer l){
        return new LAbs(l, "", DEV_CPU, 0);
    }

    layer Diff(layer l1, layer l2){
        return new LDiff(l1, l2, "", DEV_CPU, 0);
    }

    layer Diff(layer l1, float k){
        return new LDiff(l1, k, "", DEV_CPU, 0);
    }

    layer Diff(float k,layer l1){
        return new LDiff(k, l1, "", DEV_CPU, 0);
    }
    layer Div(layer l1, layer l2){
        return new LDiv(l1, l2, "", DEV_CPU, 0);
    }

    layer Div(layer l1, float k){
        return new LDiv(l1, k, "", DEV_CPU, 0);
    }

    layer Div(float k,layer l1){
        return new LDiv(k, l1, "", DEV_CPU, 0);
    }

    layer Exp(layer l){
        return new LExp(l, "", DEV_CPU, 0);
    }

    layer Log(layer l){
        return new LLog(l, "", DEV_CPU, 0);
    }

    layer Log2(layer l){
        return new LLog2(l, "", DEV_CPU, 0);
    }

    layer Log10(layer l){
        return new LLog10(l, "", DEV_CPU, 0);
    }

    layer Mult(layer l1, layer l2){
        return new LMult(l1, l2, "", DEV_CPU, 0);
    }

    layer Mult(layer l1, float k){
        return new LMult(l1, k, "", DEV_CPU, 0);
    }

    layer Mult(float k,layer l1){
        return new LMult(l1, k, "", DEV_CPU, 0);
    }
    layer Pow(layer l1, layer l2){
        return new LPow(l1, l2, "", DEV_CPU, 0);
    }

    layer Pow(layer l1, float k){
        return new LPow(l1, k, "", DEV_CPU, 0);
    }

    layer Sqrt(layer l){
        return new LSqrt(l, "", DEV_CPU, 0);
    }

    layer Sum(layer l1, layer l2){
        return new LSum(l1, l2, "", DEV_CPU, 0);
    }

    layer Sum(layer l1, float k){
        return new LSum(l1, k, "", DEV_CPU, 0);
    }

    layer Sum(float k,layer l1){
        return new LSum(l1, k, "", DEV_CPU, 0);
    }

    layer Select(layer l, vector<string> indices, string name){
        return new LSelect(l, indices, false, name, DEV_CPU, 0);
    }

    layer Permute(layer l, vector<int> dims, string name){
        return new LPermute(l, dims, name, DEV_CPU, 0);
    }

    // Reduction Layers
    layer ReduceMean(layer l, const vector<int> axis, bool keepdims){
        return new LRMean(l, axis, keepdims, "", DEV_CPU, 0);
    }

    layer ReduceVar(layer l, const vector<int> axis, bool keepdims){
        return new LRVar(l, axis, keepdims, "", DEV_CPU, 0);
    }

    layer ReduceSum(layer l, const vector<int> axis, bool keepdims){
        return new LRSum(l, axis, keepdims, "", DEV_CPU, 0);
    }

    layer ReduceMax(layer l, const vector<int> axis, bool keepdims){
        return new LRMax(l, axis, keepdims, "", DEV_CPU, 0);
    }

    layer ReduceMin(layer l, const vector<int> axis, bool keepdims){
        return new LRMin(l, axis, keepdims, "", DEV_CPU, 0);
    }


    // Generator Layers
    layer GaussGenerator(float mean, float stdev, vector<int> size){
        return new LGauss(mean, stdev, size, "", DEV_CPU, 0);
    }

    layer UniformGenerator(float low, float high, vector<int> size){
        return new LUniform(low, high, size, "", DEV_CPU, 0);
    }

    // Pooling Layers
    layer AveragePool(layer parent, const vector<int> &pool_size, const vector<int> &strides, string padding,
                      string name){
        return new LAveragePool(parent, pool_size, strides, padding, name, DEV_CPU, 0);
    }
    layer GlobalAveragePool(layer parent, string name){
        if (parent->output->ndim!=4) msg("GlobalAveragePool only over 4D tensors","GlobalAveragePool");

        int h=parent->output->shape[2];
        int w=parent->output->shape[3];
        return AveragePool(parent, {h,w},{1,1});
    }

    layer MaxPool(layer parent, const vector<int> &pool_size, const vector<int> &strides, string padding, string name){
        return new LMaxPool(parent, pool_size, strides, padding, name, DEV_CPU, 0);
    }
    layer MaxPool1D(layer parent, vector<int> pool_size, vector<int> strides, string padding, string name){

        vector<int> shape=parent->output->getShape();
        shape.push_back(1);
        LReshape *l=new LReshape(parent, shape, "", DEV_CPU, 0);

        pool_size.push_back(1);
        strides.push_back(1);
        LMaxPool *lp=new LMaxPool(l, pool_size, strides, padding, name, DEV_CPU, 0);

        vector<int> shape2=lp->output->getShape();
        shape2.pop_back();
        return new LReshape(lp,shape2, "", DEV_CPU, 0);
    }

    layer GlobalMaxPool(layer parent, string name){
        if (parent->output->ndim!=4) msg("GlobalMaxPool only over 4D tensors","GlobalMaxPool");

        int h=parent->output->shape[2];
        int w=parent->output->shape[3];
        return MaxPool(parent, {h,w}, {1,1},"none","gpool");
    }

    // Recurrent Layers

    layer RNN(layer parent, int units, string activation, bool use_bias, bool bidirectional, string name){

        return new LRNN({parent}, units, activation, use_bias, bidirectional, name, DEV_CPU, 0);
    }

    layer LSTM(layer parent, int units, bool mask_zeros, bool bidirectional, string name){
        return new LLSTM({parent}, units, mask_zeros, bidirectional, name, DEV_CPU, 0);
    }



    //////////////////////////////
    // Layers Methods
    //////////////////////////////
    void set_trainable(layer l, bool val)
    {
        l->set_trainable(val);
    }

    vlayer getOut(model net)
    {
        if (net->lout.size()) return net->lout;

        vlayer out;
        for(int i=0;i<net->layers.size();i++)
            if(net->layers[i]->child.size()==0)
                out.push_back(net->layers[i]);

        if (out.size()==0){
            throw std::runtime_error("forward over net " + net->name + " without outputs");
        }


        return out;

    }


    ////////////////////////////////////
    // Manage Tensors inside Layers
    ////////////////////////////////////

    // get COPIES of tensors
    // collect from CS when necessary
    Tensor* getOutput(layer l1){
        collectTensor(l1,"output");
        return l1->output->clone();
    }

    Tensor* getDelta(layer l1){
        collectTensor(l1,"delta");
        return l1->delta->clone();
    }

    Tensor* getParam(layer l1, int p){
        collectTensor(l1,"param",p);
        return l1->params[p]->clone();
    }

    Tensor* getGradient(layer l1,int p){
        collectTensor(l1,"gradient",p);
        return l1->gradients[p]->clone();
    }

    // get vector of tensor
    vector<Tensor*> getParams(layer l1){
      vector<Tensor*> n;
      for(int i=0;i<l1->params.size();i++) {
        collectTensor(l1,"param",i);
        n.push_back(l1->params[i]->clone());
      }
      return n;
    }

    vector<Tensor*> getGradients(layer l1){
      vector<Tensor*> n;
      for(int i=0;i<l1->gradients.size();i++) {
        collectTensor(l1,"gradients",i);
        n.push_back(l1->gradients[i]->clone());
      }
      return n;
    }


    // Copy tensors between layers
    // collect from CS when necessary
    // distribute to CS when necessary
    void copyOutput(Layer *l1,Layer *l2)
    {
        collectTensor(l1,"output");
        Tensor::copy(l1->output,l2->output);
        distributeTensor(l2,"output");
    }

    void copyDelta(Layer *l1,Layer *l2)
    {
        collectTensor(l1,"delta");
        Tensor::copy(l1->delta,l2->delta);
        distributeTensor(l2,"delta");
    }
    void copyParam(Layer *l1,Layer *l2, int p)
    {
        collectTensor(l1,"param",p);
        Tensor::copy(l1->params[p],l2->params[p]);
        distributeTensor(l2,"param",p);
    }

    void copyGradient(Layer *l1,Layer *l2, int p)
    {
        collectTensor(l1,"gradient",p);
        Tensor::copy(l1->gradients[p],l2->gradients[p]);
        distributeTensor(l2,"gradient",p);
    }



    ///////////////////////////////////////
    //  INITIALIZERS
    ///////////////////////////////////////
    layer GlorotNormal(layer l,int seed)
    {
        l->init=new IGlorotNormal(seed);
        return l;
    }

    layer GlorotUniform(layer l,int seed)
    {
        l->init=new IGlorotUniform(seed);
        return l;
    }

    layer RandomNormal(layer l, float m,float s, float seed)
    {
        l->init=new IRandomNormal(m,s,seed);
        return l;
    }

    layer RandomUniform(layer l, float min,float max, float seed)
    {
        l->init=new IRandomUniform(min,max,seed);
        return l;
    }

    layer Constant(layer l, float v)
    {
        l->init=new IConstant(v);
        return l;
    }


    ///////////////////////////////////////
    //  REGULARIZERS
    ///////////////////////////////////////
    layer L2(layer l,float l2){
        l->reg=new RL2(l2);
        return l;
    }
    layer L1(layer l,float l1){
        l->reg=new RL1(l1);
        return l;
    }
    layer L1L2(layer l,float l1,float l2){
        l->reg=new RL1L2(l1,l2);
        return l;
    }

    ///////////////////////////////////////
    //  DATASETS
    ///////////////////////////////////////
    bool exist(string name){
        if (FILE *file = fopen(name.c_str(), "r")){
            fclose(file);
            return true;
        }
        return false;
    }

    void download_dataset(string name, string ext, vector<string>link){
        string cmd;

        cout<<"Downloading "<<name<<endl;

        vector<string> file;
        file.push_back(name+"_trX."+ext);
        file.push_back(name+"_trY."+ext);
        file.push_back(name+"_tsX."+ext);
        file.push_back(name+"_tsY."+ext);

        for(int i=0;i<link.size();i++) {
          if (!exist(file[i])) {
            cout<<file[i]<<" x\n";
            cmd = "wget -q --show-progress https://www.dropbox.com/s/"+link[i]+"/"+file[i];
            int status = system(cmd.c_str());
            if (status < 0){
                msg("Error executing wget.  Is it installed?", "eddl.download_"+name);
            }
            else if (status > 0){
                msg("wget failed to download dataset (exit code: " + to_string(status) + "). See previous messages for details.", "eddl.download_"+name);
            }
          }
          else {
            cout<<file[i]<<" ✓\n";
          }
        }
      }

    void download_mnist(){
      download_dataset("mnist","bin",{"khrb3th2z6owd9t","m82hmmrg46kcugp","7psutd4m4wna2d5","q0tnbjvaenb4tjs"});
    }

    void download_cifar10(){
      download_dataset("cifar","bin",{"wap282xox5ew02d","yxhw99cu1ktiwxq","dh9vqxe9vt7scrp","gdmsve6mbu82ndp"});
    }

    void download_imdb(){
      download_dataset("imdb","bin",{"snf3vi7e1bjo8k5","c2zgsl2wb39ivlo","lkti7c12yoh18pv","cd1uocgv6abzt32"});
    }

    void download_imdb_1000(){
      download_dataset("imdb_1000","bin",{"q96yf0h84mhcbgy","jfkg2spj7bd0ca8","q2e0atxf30udvlh","wlpc9pajyvmcsiu"});
    }

    void download_drive(){
      download_dataset("drive","npy",{"sbd8eu32adcf5oi","qp0j8oiqzf6tc1a"});
    }






}//namespace
