/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <iomanip>

#include "eddl/apis/eddl.h"
#include "eddl/utils.h"
#include "eddl/serialization/onnx/eddl_onnx.h" // Not allowed
//#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/cpu/cpu_tensor.h"


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

    void setName(model m, string name)
    {
        m->name=name;
    }

    layer getLayer(model net, string lname)
    {
        return net->getLayer(lname);
    }

    void removeLayer(model net, string lname)
    {
        net->removeLayer(lname);
    }

    void initializeLayer(model net, string lname)
    {
        net->initializeLayer(lname);
    }

    void setTrainable(model net, string lname, bool val)
    {
        net->setTrainable(lname,val);
    }

    vector<vtensor> get_parameters(model net, bool deepcopy){
        return net->get_parameters(deepcopy);
    }

    void set_parameters(model net, const vector<vtensor>& params){
        net->set_parameters(params);
    }

    void build(model net, optimizer o, CompServ *cs, bool init_weights){
        // Assign default computing service
        bool do_compserv_delete = true;
        bool do_optimizer_delete = true;
        if (cs == nullptr){
            cs = new CompServ(std::thread::hardware_concurrency(), {}, {});
            do_compserv_delete = true;
        }
        if (o == nullptr){
            o = new SGD(0.001,0.9);
            do_optimizer_delete = true;
        }

        net->build(o, {}, {}, cs, init_weights, do_optimizer_delete, do_compserv_delete);
    }

    void build(model net, optimizer o, const vector<string> &lo, const vector<string> &me, CompServ *cs, bool init_weights){
        vector<Loss *> l;
        vector<Metric *> m;

        if (lo.size()!=net->lout.size()) {
            msg("Different number of losses and output layers. Use \"none\"","build");
        }

        // Replace string by functions
        for (const auto &li : lo){
            l.push_back(getLoss(li));
        }

        if (me.size()!=net->lout.size()) {
            msg("Different number of metrics and output layers. Use \"none\"","build");
        }

        for (const auto &mi : me){
            m.push_back(getMetric(mi));
        }

        // Assign default computing service
        bool do_compserv_delete = true;
        if (cs == nullptr){
            cs = new CompServ(std::thread::hardware_concurrency(), {}, {});
            do_compserv_delete = true;
        }

        // Assign default optimizer
        bool do_optimizer_delete = true;
        if (o == nullptr){
            o = new SGD(0.001,0.9);
            do_optimizer_delete = true;
        }

        net->build(o, l, m, cs, init_weights, do_optimizer_delete, do_compserv_delete);

        // do not free the objects pointed to by the elements of the following
        // vectors, but clean the internal data structure of these vectors
        m.clear();
        l.clear();
    }

    // Computing services

    // GPU
    void toGPU(model net, const string& mem){
        toGPU(net, {1}, 1, mem);
    }

    void toGPU(model net, vector<int> g, const string& mem){
        toGPU(net, g, 1, mem);
    }

    void toGPU(model net, vector<int> g, int lsb, const string& mem){
        if (mem=="low_mem") net->toGPU(g,lsb,2);
        else if (mem=="mid_mem") net->toGPU(g,lsb,1);
        else if (mem=="full_mem") net->toGPU(g,lsb,0);
        else msg("Error mem param","toGPU");
    }

    // CPU
    void toCPU(model net, int th){
        if(th==-1) { th=std::thread::hardware_concurrency(); }
        net->toCPU(th);
    }

    compserv CS_CPU(int th, const string& mem){
        if (mem=="low_mem") return new CompServ(th, {}, {}, 0, 2);
        else if (mem=="mid_mem") return new CompServ(th, {}, {}, 0, 1);
        else if (mem=="full_mem") return new CompServ(th, {}, {}, 0, 0);
        else msg("Error mem param","CS_CPU"); // Exits
        return nullptr; // To silent warnings
    }

    compserv CS_GPU(const vector<int>& g, const string& mem){
        return CS_GPU(g, 1, mem);
    }


    compserv CS_GPU(const vector<int>& g, int lsb, const string& mem){
        if (mem=="low_mem") return new CompServ(0, g, {}, lsb, 2);
        else if (mem=="mid_mem") return new CompServ(0, g, {}, lsb, 1);
        else if (mem=="full_mem") return new CompServ(0, g, {}, lsb, 0);
        else msg("Error mem param","CS_GPU"); // Exits
        return nullptr; // To silent warnings
    }

    /*compserv CS_FPGA(const vector<int> g){
      return CS_FPGA(g, 1, "full_mem");
      }
      compserv CS_FPGA(const vector<int> g, string mem){
      return CS_FPGA(g, 1, mem);
      }
      compserv CS_FPGA(const vector<int> g, int lsb){
      return CS_FPGA(g, lsb, "full_mem");
      }
      compserv CS_FPGA(const vector<int> g, int lsb, string mem){
      if (mem=="low_mem") return new CompServ(0, g, {}, lsb, 2);
      else if (mem=="mid_mem") return new CompServ(0, g, {}, lsb, 1);
      else if (mem=="full_mem") return new CompServ(0, g, {}, lsb, 0);
      else msg("Error mem param","CS_FPGA"); // Exits
      return nullptr; // To silent warnings
      }*/

    compserv CS_FPGA(const vector<int> &f,int lsb){
        return new CompServ(0, {}, f, lsb);
    }

    compserv CS_COMPSS(const string& filename){
        return new CompServ(filename);
    }

    // Info and logs
    void setlogfile(model net, const string& fname){
        net->setlogfile(fname);
    }

    void summary(model m){
        m->summary(true);
    }

    void plot(model m, const string& fname, const string& rankdir){
        m->plot(fname, rankdir);
    }

    // Serialization
    void load(model m, const string&  fname, const string& format){
        m->load(fname, format);
    }

    void save(model m, const string&  fname, const string& format){
        m->save(fname, format);
    }

    // Optimizer
    void setlr(model net,vector<float>p){
        net->setlr(p);
    }

    optimizer adadelta(float lr, float rho, float epsilon, float weight_decay){
        return new AdaDelta(lr, rho, epsilon, weight_decay);
    }

    optimizer adam(float lr, float beta_1, float beta_2, float epsilon, float weight_decay, bool amsgrad){
        return new Adam(lr, beta_1, beta_2, epsilon, weight_decay, amsgrad);
    }

    optimizer adagrad(float lr, float epsilon, float weight_decay){
        return new Adagrad(lr, epsilon, weight_decay);
    }

    optimizer adamax(float lr, float beta_1, float beta_2, float epsilon, float weight_decay){
        return new Adamax(lr, beta_1, beta_2, epsilon, weight_decay);
    }

    optimizer nadam(float lr, float beta_1, float beta_2, float epsilon, float schedule_decay){
        return new Nadam(lr, beta_1, beta_2, epsilon, schedule_decay);
    }

    optimizer rmsprop(float lr, float rho, float epsilon, float weight_decay){
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

    void evaluate(model net, const vector<Tensor *> &in, const vector<Tensor *> &out,int bs){
        net->evaluate(in, out, bs);
    }

    vector<Tensor *> predict(model m, const vector<Tensor *> &in){
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

    void show_profile() {
        __show_profile();
    }

    void reset_profile() {
        __reset_profile();
    }

    void next_batch(vector<Tensor *> in,vector<Tensor *> out){
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

    void set_quantized_mode(model net, int mode, int bits, float alpha){
        enable_quantization = mode;
        quantization_bits =  bits;
        quantization_alpha = alpha;
        net->set_quantization_mode(mode);
    }

    void end_quantization(model net){
        net->end_quantization_net();
        enable_quantization = 0;
        net->set_quantization_mode(0);
    }
    
    vlayer forward(model net,vector<Layer*> in){
        net->reset();
        net->forward(in);

        return getOut(net);
    }

    vlayer forward(model net,vector<Tensor*> in){
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
    void print_loss(model m, int batch, int mb){
        m->print_loss(batch,mb);
    }
    vector<float> get_losses(model m){
        return m->get_losses();
    }

    vector<float> get_metrics(model m){
        return m->get_metrics();
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
        if (type == "mean_squared_error" || type == "mse"){
            return new LMeanSquaredError();
        } else if (type == "categorical_cross_entropy" || type == "cross_entropy"  || type=="ce" || type=="cce"){
            return new LCategoricalCrossEntropy();
        } else if (type == "binary_cross_entropy" || type=="bce"){
            return new LBinaryCrossEntropy();
        } else if (type == "soft_cross_entropy" || type == "softmax_cross_entropy" || type == "sce"){
            return new LSoftCrossEntropy();
        } else if (type == "deprecated_cross_entropy"){
            return new LCrossEntropy();
        } else if (type == "dice"){
            return new LDice();
        } else if (type == "none"){
            return new Loss("none");
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
        else if (type == "binary_accuracy"){
            return new MBinAccuracy();
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

    layer SoftmaxDeprecated(layer parent, string name){
        show_deprecated_warning("SoftmaxDeprecated", "Softmax");
        vector<float> params = {};
        return new LActivation(parent,"softmax_deprecated", params, name, DEV_CPU, 0);
    }

    layer Softmax(layer parent, int axis, string name){
        vector<float> params = {static_cast<float>(axis)};
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
        return new LConv(parent, filters, kernel_size, strides, padding, {}, groups, dilation_rate, use_bias, name, DEV_CPU, 0);
    }

    layer Conv1D(layer parent, int filters, vector<int> kernel_size,
                 vector<int> strides, string padding,  bool use_bias,
                 int groups, vector<int> dilation_rate,string name){
        kernel_size.push_back(1);
        strides.push_back(1);
        dilation_rate.push_back(1);
        return new LConv1D(parent, filters, kernel_size, strides, padding, {}, groups, dilation_rate, use_bias, name, DEV_CPU, 0);
    }

    layer Conv2D(layer parent, int filters, const vector<int> &kernel_size,
                 const vector<int> &strides, string padding,  bool use_bias,
                 int groups, const vector<int> &dilation_rate,string name){
        return new LConv(parent, filters, kernel_size, strides, padding, {}, groups, dilation_rate, use_bias, name, DEV_CPU, 0);
    }

    layer Conv3D(layer parent, int filters, const vector<int> &kernel_size,
                 const vector<int> &strides, string padding,  bool use_bias,
                 int groups, const vector<int> &dilation_rate,string name){
        return new LConv3D(parent, filters, kernel_size, strides, padding, {}, groups, dilation_rate, use_bias, name, DEV_CPU, 0);
    }

    // Legacy
    layer PointwiseConv(layer parent, int filters,
                        const vector<int> &strides, bool use_bias,
                        int groups, const vector<int> &dilation_rate,string name){
        return new LConv(parent, filters, {1, 1}, strides, "none", {}, groups, dilation_rate, use_bias, name, DEV_CPU, 0);
    }

    layer DepthwiseConv2D(layer parent, const vector<int> &kernel_size, const vector<int> &strides, string padding, bool use_bias, const vector<int> &dilation_rate, string name){
        // A single convolutional filter is applied to each input channel
        // To achieve so, we can have one filter per channel (3xdepth) and then use groups to force each filter to have depth=1
        int filters = parent->output->shape[1];  // one filter per channel (...with depth D)
        int groups = filters;  // one filter per channel (...with depth 1)
        return new LConv(parent, filters, kernel_size, strides, "none", {}, filters, dilation_rate, use_bias, name, DEV_CPU, 0);
    }

    layer PointwiseConv2D(layer parent, int filters,
                          const vector<int> &strides, bool use_bias,
                          int groups, const vector<int> &dilation_rate,string name){
        return new LConv(parent, filters, {1, 1}, strides, "none", {}, groups, dilation_rate, use_bias, name, DEV_CPU, 0);
    }


    layer ConvT2D(layer parent, int filters, const vector<int> &kernel_size,
                  const vector<int> &strides, string padding,  bool use_bias,
                  int groups, const vector<int> &dilation_rate,string name){
        return new LConvT2D(parent, filters, kernel_size, strides, padding, {}, groups, dilation_rate, use_bias, name, DEV_CPU, 0);
    }


    layer ConvT3D(layer parent, int filters, const vector<int> &kernel_size,
                  const vector<int> &strides, string padding,  bool use_bias,
                  int groups, const vector<int> &dilation_rate,string name){
        return new LConvT3D(parent, filters, kernel_size, strides, padding, {}, groups, dilation_rate, use_bias, name, DEV_CPU, 0);
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

    layer States(const vector<int> &shape, string name){
        tshape s = vector<int>(shape.begin(), shape.end());
        if (s.size()!=2) msg("States must have two dimensions, numstates x dim_states","eddl.States");
        s.insert(s.begin(), 1); // batch + num_states + dim_states
        return new LStates(new Tensor(s), name, DEV_CPU, 0);
    }
    // Legacy
    layer UpSampling(layer parent, const vector<int> &size, string interpolation, string name){
        return UpSampling2D(parent, size, interpolation, name);
    }

    layer UpSampling2D(layer parent, const vector<int> &size, string interpolation, string name){
        // interpolation param is deprecated, we only accept "nearest"
        if (interpolation != "nearest")
            std::cerr << "Warning: In UpSampling2D the interpolation type \"" << interpolation << "\" is not valid. Only \"nearest\" is available.\n";

        const vector<int> parent_shape = parent->output->getShape();
        // Parent output must be of shape (batch, channels, height, width)
        if (parent_shape.size() != 4)
            msg("The number of dimensions of the input tensor must be 4", "EDDL::UpSampling2D");

        // Only height and width scale values
        if (size.size() != 2)
            msg("The number of dimensions of the \"size\" parameter must be 2", "EDDL::UpSampling2D");

        // Get the target output shape by applying the scale factor to the input
        vector<int> target_shape;
        for (const int i : {2, 3}) { // Loop through 2:height and 3:width dimensions (0:batch, 1:channels)
            const int dim_scale = size[i-2];
            if (dim_scale < 1)
                msg("The scale factors in \"size\" parameter must be greater or equal to 1", "EDDL::UpSampling2D");
            target_shape.push_back(parent_shape[i] * dim_scale);
        }

        return new LResize(parent, target_shape, true, getWrappingMode("nearest"), 0.0f, getTransformationMode("asymmetric"), name, DEV_CPU, 0);
    }

    layer UpSampling3D(layer parent, vector<int> new_shape, bool reshape, string da_mode, float constant, string coordinate_transformation_mode, string name){
        return new LUpSampling3D(parent, new_shape, reshape, getWrappingMode(da_mode), constant, getTransformationMode(coordinate_transformation_mode), name, DEV_CPU, 0);
    }

    layer Resize(layer parent, vector<int> new_shape, bool reshape, string da_mode, float constant, string coordinate_transformation_mode, string name){
        return new LResize(parent, new_shape, reshape, getWrappingMode(da_mode), constant, getTransformationMode(coordinate_transformation_mode), name, DEV_CPU, 0);
    }

    layer Reshape(layer parent, const vector<int> &shape, string name){
        tshape s = vector<int>(shape.begin(), shape.end());
        s.insert(s.begin(), 1);
        return new LReshape(parent, s, name, DEV_CPU, 0);
    }

    layer Flatten(layer parent, string name){
        return Reshape(parent, {-1}, name);
    }

    layer Repeat(layer parent, const vector<unsigned int>& repeats, unsigned int axis, string name){
        if(axis==-1){ axis = parent->output->ndim-1; }  // Select last dimension
        else { axis += 1; } // Batch is ignored

        return new LRepeat(parent, repeats, axis, name, DEV_CPU, 0);
    }

    layer Repeat(layer parent, unsigned int repeats, unsigned int axis, string name){
        if(axis==-1){ axis = parent->output->ndim-1; }  // Select last dimension
        else { axis += 1; } // Batch is ignored

        // Check axis values
        if(axis<0 || axis > parent->output->ndim-1){
            msg("The axis must be a number between 0 and the maximum dimension of the tensor", "Repeat");
        }
        // Repeat n times each dimension
        vector<unsigned int> vrepeats = vector<unsigned int>(parent->output->shape[axis], repeats);
        return new LRepeat(parent, vrepeats, axis, name, DEV_CPU, 0);
    }

    layer Tile(layer parent, const vector<int>& repeats, string name){
        // Inset one at the batch dimension
        vector<int> repeats_with_single_batch = repeats;
        repeats_with_single_batch.insert(repeats_with_single_batch.begin(), 1);

        // Check dimensions
        if(parent->output->ndim != repeats_with_single_batch.size()){
            msg("The number of dimensions of the output layer must match the size of 'repeats' (excluding batch)", "LTile::LTile");
        }

        return new LTile(parent, repeats_with_single_batch, name, DEV_CPU, 0);
    }

    layer Broadcast(layer parent1, layer parent2, string name){
        return new LBroadcast(parent1, parent2, name, DEV_CPU, 0);
    }

    layer Bypass(layer parent, string bypass_name, string name){
        return new LBypass(parent, bypass_name, name, DEV_CPU, 0);
    }

    layer Shape(layer parent, bool include_batch, string name){
        return new LShape(parent, include_batch, name, DEV_CPU, 0);
    }

    layer Squeeze(layer parent, const int axis, string name){
        return new LSqueeze(parent, axis, name, DEV_CPU, 0);
    }

    layer Unsqueeze(layer parent, const int axis, string name){
        return new LUnsqueeze(parent, axis, name, DEV_CPU, 0);
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

    layer ConstOfTensor(Tensor* t, string name){
        return new LConstOfTensor(t, name, DEV_CPU, 0);
    }

    layer Where(layer parent1, layer parent2, layer condition, string name){
        return new LWhere(parent1, parent2, condition, name, DEV_CPU, 0);
    }

    // Transformation Layers
    layer Shift(layer parent, vector<int> shift, string da_mode, float constant, string name){
        return new LShift(parent, shift, getWrappingMode(da_mode), constant, name, DEV_CPU, 0);
    }
    layer Rotate(layer parent, float angle, vector<int> offset_center, string da_mode, float constant, string name){
        return new LRotate(parent, angle, offset_center, getWrappingMode(da_mode), constant, name, DEV_CPU, 0);
    }

    layer Scale(layer parent, vector<int> new_shape, bool reshape, string da_mode, float constant, string coordinate_transformation_mode, string name){
        return new LScale(parent, new_shape, reshape, getWrappingMode(da_mode), constant, getTransformationMode(coordinate_transformation_mode), name, DEV_CPU, 0);
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

    layer Pad(layer parent, vector<int> pads, float constant, string name){
        return new LPad(parent, pads, constant, name, DEV_CPU, 0);
    }

    // Data augmentation Layers
    layer RandomShift(layer parent, vector<float> factor_x, vector<float> factor_y, string da_mode, float constant, string name){
        return new LShiftRandom(parent, factor_x, factor_y, getWrappingMode(da_mode), constant, name, DEV_CPU, 0);
    }

    layer RandomRotation(layer parent, vector<float> factor, vector<int> offset_center, string da_mode, float constant, string name){
        return new LRotateRandom(parent, factor, offset_center, getWrappingMode(da_mode), constant, name, DEV_CPU, 0);
    }

    layer RandomScale(layer parent, vector<float> factor, string da_mode, float constant, string coordinate_transformation_mode, string name){
        return new LScaleRandom(parent, factor, getWrappingMode(da_mode), constant, getTransformationMode(coordinate_transformation_mode), name, DEV_CPU, 0);
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
        // Expand dimension if needed
        if(parent->output->shape.size()==3){ parent = _expand3d_to_4d(parent, "BatchNormalization"); }

        return new LBatchNorm(parent, momentum, epsilon, affine, name, DEV_CPU, 0);
    }

    layer BatchNormalization(layer parent, bool affine, float momentum, float epsilon,  string name){
        // Expand dimension if needed
        if(parent->output->shape.size()==3){ parent = _expand3d_to_4d(parent, "BatchNormalization"); }

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

  layer Transform(layer parent, int copy_cpu_to_fpga, int copy_fpga_to_cpu, int transform, int mode, string name) {
      return new LTransform(parent, copy_cpu_to_fpga, copy_fpga_to_cpu, transform, name, DEV_CPU, 0);
  }

    //  Operator Layers
    layer Abs(layer l){
        return new LAbs(l, "", DEV_CPU, 0);
    }


    layer Sub(layer l1, layer l2){
        return new LDiff(l1, l2, "", DEV_CPU, 0);
    }

    layer Sub(layer l1, float k){
        return new LDiff(l1, k, "", DEV_CPU, 0);
    }

    layer Sub(float k,layer l1){
        return new LDiff(k, l1, "", DEV_CPU, 0);
    }

    layer Diff(layer l1, layer l2){
        show_deprecated_warning("Diff", "Sub");
        return new LDiff(l1, l2, "", DEV_CPU, 0);
    }

    layer Diff(layer l1, float k){
        show_deprecated_warning("Diff", "Sub");
        return new LDiff(l1, k, "", DEV_CPU, 0);
    }

    layer Diff(float k,layer l1){
        show_deprecated_warning("Diff", "Sub");
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

    layer Clamp(layer l, float min, float max, string name){
        return new LClamp(l, min, max, name, DEV_CPU, 0);
    }

    layer Clip(layer l, float min, float max, string name){
        return Clamp(l, min, max, name); // Alias for clamp
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

    layer Pow(layer l1, float k){
        return new LPow(l1, k, "", DEV_CPU, 0);
    }

    layer Sqrt(layer l){
        return new LSqrt(l, "", DEV_CPU, 0);
    }

    layer Add(layer l1, layer l2){
        return new LSum(l1, l2, "", DEV_CPU, 0);
    }

    layer Add(layer l1, float k){
        return new LSum(l1, k, "", DEV_CPU, 0);
    }

    layer Add(float k,layer l1){
        return new LSum(l1, k, "", DEV_CPU, 0);
    }

    layer Sum(layer l1, layer l2){
        show_deprecated_warning("Sum", "Add");
        return new LSum(l1, l2, "", DEV_CPU, 0);
    }

    layer Sum(layer l1, float k){
        show_deprecated_warning("Sum", "Add");
        return new LSum(l1, k, "", DEV_CPU, 0);
    }

    layer Sum(float k,layer l1){
        show_deprecated_warning("Sum", "Add");
        return new LSum(l1, k, "", DEV_CPU, 0);
    }

    layer Select(layer l, vector<string> indices, string name){
        return new LSelect(l, indices, name, DEV_CPU, 0);
    }

    layer Slice(layer l, vector<string> indices, string name){
        return new LSelect(l, indices, name, DEV_CPU, 0);
    }

    layer Expand(layer l, int size, string name){
        return new LExpand(l, size, name, DEV_CPU, 0);
    }

    layer Permute(layer l, vector<int> dims, string name){
        return new LPermute(l, dims, name, DEV_CPU, 0);
    }

    vlayer Split(layer l, vector<int> indexes, int axis, bool merge_sublayers, string name){
        auto* sl = new LSplit(l, indexes, axis, merge_sublayers, name, DEV_CPU, 0);
        return  sl->split_layers;
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

    layer ReduceArgMax(layer l, vector<int> axis, bool keepdims){
        return new LRArgmax(l, axis, keepdims, "", DEV_CPU, 0);
    }

    // Generator Layers
    layer GaussGenerator(float mean, float stdev, vector<int> size){
        return new LGauss(mean, stdev, size, "", DEV_CPU, 0);
    }

    layer UniformGenerator(float low, float high, vector<int> size){
        return new LUniform(low, high, size, "", DEV_CPU, 0);
    }

    // Generic (in-theory)
    layer MaxPool(layer parent, const vector<int> &pool_size, const vector<int> &strides, string padding, string name){
        return new LMaxPool(parent, pool_size, strides, padding, name, DEV_CPU, 0);
    }

    layer MaxPool1D(layer parent, vector<int> pool_size, vector<int> strides, string padding, string name){
        pool_size.push_back(1);
        strides.push_back(1);
        return new LMaxPool1D(parent, pool_size, strides, padding, name, DEV_CPU, 0);
    }

    layer MaxPool2D(layer parent, vector<int> pool_size, vector<int> strides, string padding, string name){
        return new LMaxPool(parent, pool_size, strides, padding, name, DEV_CPU, 0);
    }

    layer MaxPool3D(layer parent, vector<int> pool_size, vector<int> strides, string padding, string name){
        return new LMaxPool3D(parent, pool_size, strides, padding, name, DEV_CPU, 0);
    }

    // Pooling Layers
    layer AveragePool(layer parent, const vector<int> &pool_size, const vector<int> &strides, string padding, string name){
        return new LAveragePool(parent, pool_size, strides, padding, name, DEV_CPU, 0);
    }
    layer AvgPool(layer parent, const vector<int> &pool_size, const vector<int> &strides, string padding, string name){
        return  AveragePool(parent, pool_size, strides, padding, name);
    }

    layer AveragePool1D(layer parent, vector<int> pool_size, vector<int> strides, string padding, string name){
        pool_size.push_back(1);
        strides.push_back(1);
        return new LAveragePool1D(parent, pool_size, strides, padding, name, DEV_CPU, 0);
    }
    layer AvgPool1D(layer parent, vector<int> pool_size, vector<int> strides, string padding, string name){
        return  AveragePool1D(parent, pool_size, strides, padding, name);
    }

    layer AveragePool2D(layer parent, vector<int> pool_size, vector<int> strides, string padding, string name){
        return new LAveragePool(parent, pool_size, strides, padding, name, DEV_CPU, 0);
    }

    layer AvgPool2D(layer parent, vector<int> pool_size, vector<int> strides, string padding, string name){
        return  AveragePool2D(parent, pool_size, strides, padding, name);
    }

    layer AveragePool3D(layer parent, vector<int> pool_size, vector<int> strides, string padding, string name){
        return new LAveragePool3D(parent, pool_size, strides, padding, name, DEV_CPU, 0);
    }

    layer AvgPool3D(layer parent, vector<int> pool_size, vector<int> strides, string padding, string name){
        return  AveragePool3D(parent, pool_size, strides, padding, name);
    }


    layer GlobalMaxPool(layer parent, string name){
        // Expand dimension if needed
        if(parent->output->shape.size()==3){ parent = _expand3d_to_4d(parent, "GlobalMaxPool"); }

        return GlobalMaxPool2D(parent, name);
    }

    layer GlobalMaxPool1D(layer parent, string name){
        if (parent->output->ndim!=3) msg("GlobalMaxPool1D only works over 3D tensors","GlobalMaxPool1D");

        int h=parent->output->shape[2];

        if (name.empty()) { name = "GlobalMaxPool1D"; }  // Set default name
        return MaxPool1D(parent, {h}, {1}, "none", name);
    }

    layer GlobalMaxPool2D(layer parent, string name){
        // Check dimension
        if (parent->output->ndim!=4) msg("GlobalMaxPool only over 4D tensors","GlobalMaxPool2D");

        int h=parent->output->shape[2];
        int w=parent->output->shape[3];

        if(name.empty()) { name = "GlobalMaxPool2D"; }  // Set default name
        return MaxPool(parent, {h,w}, {1,1},"none", name);
    }

    layer GlobalMaxPool3D(layer parent, string name){
        // Check dimension
        if (parent->output->ndim!=5) msg("GlobalMaxPool only works over 5D tensors","GlobalMaxPool3D");

        int d=parent->output->shape[2];
        int h=parent->output->shape[3];
        int w=parent->output->shape[4];

        if(name.empty()) { name = "GlobalMaxPool3D"; }  // Set default name
        return MaxPool3D(parent, {d, h,w}, {1, 1,1},"none", name);
    }

    layer GlobalAveragePool(layer parent, string name){
        // Expand dimension if needed
        if(parent->output->shape.size()==3){ parent = _expand3d_to_4d(parent, "GlobalAveragePool"); }

        return GlobalAveragePool2D(parent, name);
    }
    layer GlobalAvgPool(layer parent, string name){
        return GlobalAveragePool(parent, name);
    }

    layer GlobalAveragePool1D(layer parent, string name){
        // Expands dimensions if needed
        if (parent->output->ndim!=3) msg("GlobalAveragePool1D only works over 3D tensors","GlobalAveragePool1D");

        int h=parent->output->shape[2];

        if(name.empty()) { name = "GlobalAveragePool1D"; }  // Set default name
        return AveragePool1D(parent, {h},{1}, "none", name);
    }
    layer GlobalAvgPool1D(layer parent, string name){
        return GlobalAveragePool1D(parent, name);
    }

    layer GlobalAveragePool2D(layer parent, string name){
        // Check dimension
        if (parent->output->ndim!=4) msg("GlobalAveragePool only works over 4D tensors","GlobalAveragePool2D");

        int h=parent->output->shape[2];
        int w=parent->output->shape[3];

        if(name.empty()) { name = "GlobalAveragePool2D"; }  // Set default name
        return AveragePool(parent, {h,w},{1,1},  "none",name);
    }
    layer GlobalAvgPool2D(layer parent, string name){
        return GlobalAveragePool2D(parent, name);
    }

    layer GlobalAveragePool3D(layer parent, string name){
        // Expands dimensions if needed
        if (parent->output->ndim!=5) msg("GlobalAveragePool3D only works over 5D tensors","GlobalAveragePool3D");

        int d=parent->output->shape[2];
        int h=parent->output->shape[3];
        int w=parent->output->shape[4];

        if(name.empty()) { name = "GlobalAveragePool3D"; }  // Set default name
        return AveragePool3D(parent, {d,h,w},{1,1,1}, "none", name);
    }

    layer GlobalAvgPool3D(layer parent, string name){
        return GlobalAveragePool3D(parent, name);
    }

    // Recurrent Layers
    layer RNN(layer parent, int units, string activation, bool use_bias, bool bidirectional, string name){

        return new LRNN({parent}, units, activation, use_bias, bidirectional, name, DEV_CPU, 0);
    }

    layer LSTM(layer parent, int units, bool mask_zeros, bool bidirectional, string name){
        return new LLSTM({parent}, units, mask_zeros, bidirectional, name, DEV_CPU, 0);
    }

    layer LSTM(vector<layer> parent, int units, bool mask_zeros, bool bidirectional, string name){
        return new LLSTM(parent, units, mask_zeros, bidirectional, name, DEV_CPU, 0);
    }

    layer GRU(layer parent, int units, bool mask_zeros, bool bidirectional, string name) {
        return new LGRU({parent}, units, mask_zeros, bidirectional, name, DEV_CPU, 0);
    }

    layer GRU(vector<layer> parent, int units, bool mask_zeros, bool bidirectional, string name) {
        return new LGRU(parent, units, mask_zeros, bidirectional, name, DEV_CPU, 0);
    }

    layer GetStates(layer parent)
    {
        return new LCopyStates({parent},"getstates", DEV_CPU,0);
    }


    bool isrec(layer l)
    {
        if (l->isrecurrent) return true;

        bool rec=false;
        for(int i=0;i<l->parent.size();i++) {
            if (l->parent[i]->isrecurrent) {rec=true;break;}
            else {
                if (isrec(l->parent[i])) {rec=true; break;}
            }
        }

        return rec;
    }

    void setDecoder(layer l)
    {

        l->isdecoder=true;

        int p=l->child.size();
        for(int i=0;i<p;i++)
            setDecoder(l->child[i]);

    }


    //////////////////////////////
    // Layers Methods
    //////////////////////////////

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
        Net *n=l1->net;
        if (n->rnet==nullptr) {
            //cout<<"collect Output from layer "<<l1->name<<endl;
            collectTensor(l1,"output");
            return l1->output->clone();  // Why not return addresses so that we can easily avoid potential memory leaks?
        }
        else {
            //cout<<"get output from recurrent"<<endl;
            int length=0;

            vector<int> shape;
            for(auto l: n->rnet->layers)
                if (l->sorig==l1) {
                    if (length==0) shape=l->output->shape;
                    length++;
                }

            shape.insert(shape.begin(), length);
            // length x batch x dim_layer
            Tensor *output=new Tensor(shape,DEV_CPU);

            int i=0;
            for(auto l: n->rnet->layers)
                if (l->sorig==l1) {
                    Tensor *out=getOutput(l);
                    shape[0]=1;
                    out->reshape_(shape);

                    // put into output
                    vector<string> s;
                    s.push_back(to_string(i));
                    for(int j=1;j<shape.size();j++) s.push_back(":");

                    output->set_select(s, out);

                    s.clear();
                    i++;
                    delete out;
                }
            return output;
        }
    }



    // get COPIES of tensors
    // collect from CS when necessary
    Tensor* getInput(layer l1){
        Net *n=l1->net;
        if (n->rnet==nullptr) {
            //cout<<"collect Output from layer "<<l1->name<<endl;
            collectTensor(l1,"input");
            return l1->output->clone();  // Why not return addresses so that we can easily avoid potential memory leaks?
        }
        else {
            cout<<"get input from recurrent"<<endl;
            int length=0;

            vector<int> shape;
            for(auto l: n->rnet->layers)
                if (l->sorig==l1) {
                    if (length==0) shape=l->output->shape;
                    length++;
                }

            shape.insert(shape.begin(), length);
            // length x batch x dim_layer
            Tensor *output=new Tensor(shape,DEV_CPU);

            int i=0;
            for(auto l: n->rnet->layers)
                if (l->sorig==l1) {
                    Tensor *out=getOutput(l);
                    shape[0]=1;
                    out->reshape_(shape);

                    // put into output
                    vector<string> s;
                    s.push_back(to_string(i));
                    for(int j=1;j<shape.size();j++) s.push_back(":");

                    output->set_select(s, out);

                    s.clear();
                    i++;
                    delete out;
                }
            return output;
        }
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

    Tensor* getState(layer l1,int p){
        collectTensor(l1,"state",p);
        return l1->states[p]->clone();
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

    vector<Tensor*> getStates(layer l1){
        vector<Tensor*> n;
        for(int i=0;i<l1->states.size();i++) {
            collectTensor(l1,"state",i);
            n.push_back(l1->states[i]->clone());
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
        if (p==-1) {
            cout<<"copy all params from "<<l1->name<<" to "<<l2->name<<endl;
            for(int i=0;i<l1->params.size();i++) {
                collectTensor(l1,"param",i);
                Tensor::copy(l1->params[i],l2->params[i]);
                distributeTensor(l2,"param",i);
            }
        }
        else {
            collectTensor(l1,"param",p);
            Tensor::copy(l1->params[p],l2->params[p]);
            distributeTensor(l2,"param",p);
        }
    }

    void copyGradient(Layer *l1,Layer *l2, int p)
    {
        collectTensor(l1,"gradient",p);
        Tensor::copy(l1->gradients[p],l2->gradients[p]);
        distributeTensor(l2,"gradient",p);
    }

    void distributeParams(Layer *l)
    {
        for(int i=0;i<l->params.size();i++)
            distributeTensor(l,"param",i);
    }



    ///////////////////////////////////////
    //  INITIALIZERS
    ///////////////////////////////////////
    layer GlorotNormal(layer l,int seed)
    {
        if (l->init != nullptr) delete l->init;
        l->init = new IGlorotNormal(seed);
        return l;
    }

    layer HeUniform(layer l, int seed)
    {
        if (l->init != nullptr) delete l->init;
        l->init = new IHeUniform(seed);
        return l;
    }
    layer HeNormal(layer l, int seed)
    {
        if (l->init != nullptr) delete l->init;
        l->init = new IHeNormal(seed);
        return l;
    }

    layer GlorotUniform(layer l, int seed)
    {
        if (l->init != nullptr) delete l->init;
        l->init = new IGlorotUniform(seed);
        return l;
    }

    layer RandomNormal(layer l, float m, float s, float seed)
    {
        if (l->init != nullptr) delete l->init;
        l->init = new IRandomNormal(m, s, seed);
        return l;
    }

    layer RandomUniform(layer l, float min, float max, float seed)
    {
        if (l->init != nullptr) delete l->init;
        l->init = new IRandomUniform(min, max, seed);
        return l;
    }

    layer Constant(layer l, float v)
    {
        if (l->init != nullptr) delete l->init;
        l->init = new IConstant(v);
        return l;
    }


    ///////////////////////////////////////
    //  REGULARIZERS
    ///////////////////////////////////////
    layer L2(layer l,float l2){
        if (l->reg != nullptr) delete l->reg;
        l->reg = new RL2(l2);
        return l;
    }
    layer L1(layer l,float l1){
        if (l->reg != nullptr) delete l->reg;
        l->reg = new RL1(l1);
        return l;
    }
    layer L1L2(layer l, float l1, float l2){
        if (l->reg != nullptr) delete l->reg;
        l->reg = new RL1L2(l1, l2);
        return l;
    }


    bool exist(string name){
        if (FILE *file = fopen(name.c_str(), "r")){
            fclose(file);
            return true;
        }
        return false;
    }

    ///////////////////////////////////////
    //  Pretrained Models
    ///////////////////////////////////////

    void download_model(string name,string link)
    {
        string cmd;
        cout<<"Downloading "<<name<<endl;

        if (!exist(name)) {
            cout<<name<<" x\n";
            cmd = "wget -q  https://www.dropbox.com/s/"+link+"/"+name;
            int status = system(cmd.c_str());
            if (status < 0){
                msg("Error executing wget.  Is it installed?", "eddl.download_"+name);
            }
            else if (status > 0){
                cout<<cmd<<endl;
                msg("wget failed to download dataset (exit code: " + to_string(status) + "). See previous messages for details.", "eddl.download_"+name);
            }
        }
        else {
            cout<<name<<" ✓\n";
        }
    }

    Net* download_vgg16(bool top, vector<int> input_shape)
    {
        download_model("vgg16.onnx","2ovxkt64y11c083");

        Net *net;

        cout<<"Import ONNX..."<<endl;

        if (input_shape.size())
            net = import_net_from_onnx_file("vgg16.onnx", input_shape, DEV_CPU);
        else
            net = import_net_from_onnx_file("vgg16.onnx", DEV_CPU);

        Layer *l=getLayer(net,"data"); l->name="input";
        if (top) {
            net->removeLayer("vgg0_dense2_fwd");
            net->removeLayer("vgg0_dropout1_fwd");
            net->removeLayer("vgg0_dense1_relu_fwd");
            net->removeLayer("vgg0_dense1_fwd");
            net->removeLayer("vgg0_dropout0_fwd");
            net->removeLayer("vgg0_dense0_relu_fwd");
            net->removeLayer("vgg0_dense0_fwd");

            Layer *l=getLayer(net,"flatten_60"); l->name="top";

        }

        return net;
    }

    Net* download_vgg16_bn(bool top, vector<int> input_shape)
    {
        download_model("vgg16bn-7.onnx", "df7nd1ydsba9yp5");

        cout << "Import ONNX..." << endl;

        Net *net;
        if (input_shape.size())
            net = import_net_from_onnx_file("vgg16bn-7.onnx", input_shape, DEV_CPU);
        else
            net = import_net_from_onnx_file("vgg16bn-7.onnx", DEV_CPU);

        Layer *l = getLayer(net, "data"); l->name = "input";
        if (top) {
            net->removeLayer("vgg0_dense2_fwd");
            net->removeLayer("flatten_135");
            net->removeLayer("vgg0_dropout1_fwd");
            net->removeLayer("vgg0_dense1_relu_fwd");
            net->removeLayer("vgg0_dense1_fwd");
            net->removeLayer("flatten_130");
            net->removeLayer("vgg0_dropout0_fwd");
            net->removeLayer("vgg0_dense0_relu_fwd");
            net->removeLayer("vgg0_dense0_fwd");

            Layer *l = getLayer(net, "flatten_125"); l->name = "top";
        }

        return net;
    }

    Net* download_vgg19(bool top, vector<int> input_shape)
    {
        download_model("vgg19-7.onnx", "859b3zsr4dvvr26");

        cout << "Import ONNX..." << endl;

        Net *net;
        if (input_shape.size())
            net = import_net_from_onnx_file("vgg19-7.onnx", input_shape, DEV_CPU);
        else
            net = import_net_from_onnx_file("vgg19-7.onnx", DEV_CPU);

        Layer *l = getLayer(net, "data"); l->name = "input";
        if (top) {
            net->removeLayer("vgg0_dense2_fwd");
            net->removeLayer("flatten_82");
            net->removeLayer("vgg0_dropout1_fwd");
            net->removeLayer("vgg0_dense1_relu_fwd");
            net->removeLayer("vgg0_dense1_fwd");
            net->removeLayer("flatten_77");
            net->removeLayer("vgg0_dropout0_fwd");
            net->removeLayer("vgg0_dense0_relu_fwd");
            net->removeLayer("vgg0_dense0_fwd");

            Layer *l = getLayer(net, "flatten_72"); l->name = "top";
        }

        return net;
    }

    Net* download_vgg19_bn(bool top, vector<int> input_shape)
    {
        download_model("vgg19bn-7.onnx", "93ha0nilvin38rz");

        cout << "Import ONNX..." << endl;

        Net *net;
        if (input_shape.size())
            net = import_net_from_onnx_file("vgg19bn-7.onnx", input_shape, DEV_CPU);
        else
            net = import_net_from_onnx_file("vgg19bn-7.onnx", DEV_CPU);

        Layer *l = getLayer(net, "data"); l->name = "input";
        if (top) {
            net->removeLayer("vgg0_dense2_fwd");
            net->removeLayer("flatten_162");
            net->removeLayer("vgg0_dropout1_fwd");
            net->removeLayer("vgg0_dense1_relu_fwd");
            net->removeLayer("vgg0_dense1_fwd");
            net->removeLayer("flatten_157");
            net->removeLayer("vgg0_dropout0_fwd");
            net->removeLayer("vgg0_dense0_relu_fwd");
            net->removeLayer("vgg0_dense0_fwd");

            Layer *l = getLayer(net, "flatten_152"); l->name = "top";
        }

        return net;
    }

    Net* download_resnet18(bool top, vector<int> input_shape)
    {
        download_model("resnet18.onnx","re7jodd12srksd7");
        Net *net;

        cout<<"Import ONNX..."<<endl;

        if (input_shape.size())
            net = import_net_from_onnx_file("resnet18.onnx", input_shape, DEV_CPU);
        else
            net = import_net_from_onnx_file("resnet18.onnx", DEV_CPU);

        Layer *l=getLayer(net,"data"); l->name="input";
        if (top) {
            net->removeLayer("resnetv15_dense0_fwd");
            Layer *l=getLayer(net,"flatten_170"); l->name="top";

        }

        return net;
    }

    Net* download_resnet34(bool top, vector<int> input_shape)
    {
        download_model("resnet34.onnx","ikcaak0q2cee8k1");
        Net *net;

        cout<<"Import ONNX..."<<endl;

        if (input_shape.size())
            net = import_net_from_onnx_file("resnet34.onnx", input_shape, DEV_CPU);
        else net = import_net_from_onnx_file("resnet34.onnx", DEV_CPU);

        Layer *l=getLayer(net,"data"); l->name="input";
        if (top) {
            net->removeLayer("resnetv16_dense0_fwd");
            Layer *l=getLayer(net,"flatten_306"); l->name="top";

        }

        return net;
    }

    Net* download_resnet50(bool top, vector<int> input_shape)
    {
        download_model("resnet50.onnx","hg4r3z8m6wsnwk3");
        Net *net;

        cout<<"Import ONNX..."<<endl;

        if (input_shape.size())
            net = import_net_from_onnx_file("resnet50.onnx", input_shape, DEV_CPU);
        else net = import_net_from_onnx_file("resnet50.onnx", DEV_CPU);

        Layer *l=getLayer(net,"data"); l->name="input";
        if (top) {
            net->removeLayer("resnetv17_dense0_fwd");
            Layer *l=getLayer(net,"flatten_473"); l->name="top";

        }

        return net;
    }

    Net* download_resnet101(bool top, vector<int> input_shape)
    {
        download_model("resnet101.onnx","eaxjju4ftrwoti0");
        Net *net;

        cout<<"Import ONNX..."<<endl;

        if (input_shape.size())
            net = import_net_from_onnx_file("resnet101.onnx", input_shape, DEV_CPU);
        else net = import_net_from_onnx_file("resnet101.onnx", DEV_CPU);

        Layer *l=getLayer(net,"data"); l->name="input";
        if (top) {
            net->removeLayer("resnetv18_dense0_fwd");
            Layer *l=getLayer(net,"flatten_932"); l->name="top";

        }

        return net;
    }

    Net* download_resnet152(bool top, vector<int> input_shape)
    {
        download_model("resnet152.onnx","phoffbhgnolg95u");
        Net *net;

        cout<<"Import ONNX..."<<endl;

        if (input_shape.size())
            net = import_net_from_onnx_file("resnet152.onnx", input_shape, DEV_CPU);
        else net = import_net_from_onnx_file("resnet152.onnx", DEV_CPU);

        Layer *l=getLayer(net,"data"); l->name="input";
        if (top) {
            net->removeLayer("resnetv19_dense0_fwd");
            Layer *l=getLayer(net,"flatten_1391"); l->name="top";

        }

        return net;
    }

    Net* download_densenet121(bool top, vector<int> input_shape)
    {
        download_model("densenet121.onnx","mod7a1pf0eldyd1");
        Net *net;
        cout<<"Import ONNX..."<<endl;

        if (input_shape.size())
            net = import_net_from_onnx_file("densenet121.onnx", input_shape, DEV_CPU);
        else net = import_net_from_onnx_file("densenet121.onnx", DEV_CPU);

        Layer *l=getLayer(net,"data_0"); l->name="input";
        if (top) {
            net->removeLayer("conv2d121");
            Layer *l=getLayer(net,"avgpool10");
            l->name="top";
        }

        return net;
    }


    ///////////////////////////////////////
    //  DATASETS
    ///////////////////////////////////////
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
                cmd = "wget -q  https://www.dropbox.com/s/"+link[i]+"/"+file[i];
                int status = system(cmd.c_str());
                if (status < 0){
                    msg("Error executing wget.  Is it installed?", "eddl.download_"+name);
                }
                else if (status > 0){
                    cout<<cmd<<endl;
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

    void download_imdb_2000(){
        download_dataset("imdb_2000","bin",{"4m0h8ep53mixq6x","zekpjclm58tdevk","1bgdr8mz1lqkhgi","6cwob77654lruwq"});
    }

    void download_eutrans(){
        download_dataset("eutrans","bin",{"2w0p7f4un6ci94v","g4k1bc6p4bow9tf","egcfin16gl9t92y","n8ks3lyqyhxx1e8"});
    }

    void download_flickr(){
        download_dataset("flickr","bin",{"452pyxe9x5jpnwb","24c2d5bm6pug8gg"});
    }

    void download_drive(){
        download_dataset("drive","bin",{"tf3uzrsjtv4jiey","xakcuhby30ylpes"});
    }

    ///////////////////////////////////////
    //  HLSinf accelerators
    ///////////////////////////////////////
    void download_hlsinf(int version, int subversion){
        char file[200];

        sprintf(file, "hlsinf_v%0d.%0d.xclbin", version, subversion);

        cout << "Downloading " << file << endl;

        if (!exist(file)) {
            char cmd[200];
            sprintf(cmd, "wget -q https://www.dropbox.com/s/%s", file);
            int status = system(cmd);
            if (status < 0){
                msg("Error executing wget.  Is it installed?", "eddl.download_hlsinf");
            }
            else if (status > 0){
                cout<<cmd<<endl;
                msg("wget failed to download HLSinf accelerator (exit code: " + to_string(status) + ").", "eddl.download_hlsinf");
            }
        }
    }


    // Auxiliar functions
    layer _expand3d_to_4d(layer parent, string name){
        layer p = parent;
        if(parent->output->shape.size()==3){
            std::cerr << name << " only works over 2D or 4D tensors. Since a 3D tensor was received, its shape was automatically unsqueezed to a 4D tensor." << std::endl;
            std::cerr << "()" << std::endl;
            p = Unsqueeze(p, 0);  // ([Batch - ignored], d0, d1)
        }
        return p;
    }

    vector<string> read_txt_file(const string& filename){
        return read_lines_from_file(filename);  // I had to change the name of the function. Utils should have a namespace
    }

    string get_topk_predictions(Tensor* class_probs, const vector<string>& class_names, int k, int decimals){
        // Check tensor dimensions
        if (class_probs == nullptr || class_probs->size == 0){
            throw std::runtime_error("The given class probability tensor is empty");

        }
        // Check class names sizes
        if(class_probs->size!=class_names.size()){
            throw std::runtime_error("The number of elements in the class probability tensor does not match the number of class names");
        }

        // Check K
        if(k <= 0  || k >= class_probs->size) {
            throw std::runtime_error("'k' must be a number greater than zero and smaller than the number of classes");
        }

        // Sort indices by probability
        Tensor* top_k_probs_idx = class_probs->argsort(true);
        std::stringstream stream;
        for(int i=0; i<k; i++){
            int idx = (int)top_k_probs_idx->ptr[i];
            float prob = class_probs->ptr[idx] * 100.0f;
            stream << i+1 << ". " << class_names[idx] << " (" << std::fixed << std::setprecision(decimals) << prob << "%)" << std::endl;
        }
        std::string result = stream.str();
        return result;
    }

}
