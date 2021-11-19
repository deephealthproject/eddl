/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_H
#define EDDL_H

#include <vector>
#include <thread>
//#include <pthread.h>
#include <functional>

#include "eddl/net/net.h"
#include "eddl/net/netloss.h"
#include "eddl/initializers/initializer.h"
#include "eddl/regularizers/regularizer.h"
#include "eddl/losses/loss.h"
#include "eddl/metrics/metric.h"

#include "eddl/layers/layer.h"
#include "eddl/layers/conv/layer_conv.h"
#include "eddl/layers/core/layer_core.h"
#include "eddl/layers/auxiliar/layer_auxiliar.h"
#include "eddl/layers/da/layer_da.h"
#include "eddl/layers/fused/layer_fused.h"
#include "eddl/layers/generators/layer_generators.h"
#include "eddl/layers/merge/layer_merge.h"
#include "eddl/layers/noise/layer_noise.h"
#include "eddl/layers/normalization/layer_normalization.h"
#include "eddl/layers/operators/layer_operators.h"
#include "eddl/layers/reductions/layer_reductions.h"
#include "eddl/layers/pool/layer_pool.h"
#include "eddl/layers/recurrent/layer_recurrent.h"


// EDDL namespace defines the API
namespace eddl {

    typedef Layer* layer;
    typedef Net* model;
    typedef Optimizer* optimizer;
    typedef Initializer* initializer;
    typedef Regularizer* regularizer;
    typedef CompServ* compserv;
    typedef NetLoss * loss;
    typedef NetLoss * metric;

    ///////////////////////////////////////
    //  MODEL METHODS
    ///////////////////////////////////////

    // Creation
    /**
      *  @brief Instantiates model, taking two vectors, one of input layers and another of output layers.
      *
      *  @param in  Vector with model input layers, typically Input({...})
      *  @param out  Vector with the ouputs of the model. Example: {Sigmoid(MyModel())}
      *  @return     Model instance
    */
    model Model(vlayer in, vlayer out);
    void setName(model m, string name);
    layer getLayer(Net *net, string l);
    void removeLayer(Net *net, string l);
    void initializeLayer(Net *net, string l);
    void setTrainable(model net, string lanme, bool val);

    /**
      *  @brief Return a vector of vector of tensors with the parameters of each layer.
      *  These tensors are in CPU. Function transparent for distributed mode
      *
      *  @param net  Model
      *  @param deepcopy  Whether the return vectors contains reference to the tensors or a copy of them
      *  @return Vector of Vector of Tensors
    */
    vector<vtensor> get_parameters(model net, bool deepcopy=false);

    /**
    *  @brief Sets the parameters of the net.
     *  Function transparent for distributed mode
    *
    *  @param net  Model
    *  @param params  Params to copy to the net
    */
    void set_parameters(model net, const vector<vtensor>& params);

    /**
      *  @brief Configures the model for training.
      *
      *  @param net  Model
      *  @param o  Optimizer
      *  @param cs  Computing service
      *  @param init_weights  'True' if the weights can be initialized to random values, else False (e.g.: Used when loading a pretrained model)
      *  @return     (void)
    */
    void build(model net, optimizer o=nullptr, CompServ *cs=nullptr, bool init_weigths=true);

    /**
      *  @brief Configures the model for training.
      *
      *  @param net  Model
      *  @param o  Optimizer
      *  @param lo  Vector with losses
      *  @param me  Vector with metrics
      *  @param cs  Computing service
      *  @param init_weights  'True' if the weights can be initialized to random values, else False (e.g.: Used when loading a pretrained model)
      *  @return     (void)
    */
    void build(model net, optimizer o, const vector<string> &lo, const vector<string> &me, CompServ *cs=nullptr, bool init_weights=true);

    // Computing services



    /**
      *  @brief Assign model operations to the CPU.
      *  @param net  Model
      *  @param th  CPU Threads. (if '-1', use all threads)
      *  @return     (void)
    */
    void toCPU(model net, int th=-1);


    /**
      *  @brief Assign model operations to the GPU.
      *  @param net  Model
      *  @param g  Vector of bools to set which GPUs will be used (1=on, 0=off)
      *  @param mem  Indicates the memory consumption of the model. One of "full_mem" (default), "mid_mem" or "low_mem".
      *  @return     (void)
    */
    void toGPU(model net, vector<int> g={1}, int lsb=1, const string& mem="full_mem");
    void toGPU(model net, const string& mem="full_mem");
    void toGPU(model net, vector<int> g={1}, const string& mem="full_mem");

    /**
      *  @brief Executes the code in the CPU.
      *
      *  @param th  CPU Threads. (if '-1', use all threads)
      *  @param mem  Indicates the memory consumption of the model. One of "full_mem" (default), "mid_mem" or "low_mem".
      *  @return     The computer service itself.
    */
    compserv CS_CPU(int th=-1, const string& mem="full_mem");


    /**
      *  @brief Executes the code in the GPU.
      *
      *  @param g  Vector of bools to set which GPUs will be used (1=on, 0=off)
      *  @param mem  Indicates the memory consumption of the model. One of "full_mem" (default), "mid_mem" or "low_mem".
      *  @return     The computer service itself.
    */
    compserv CS_GPU(const vector<int>& g, const string& mem="full_mem");

    /**
      *  @brief Executes the code in the GPU.
      *
      *  @param g  Vector of bools to set which GPUs will be used (1=on, 0=off)
      *  @param lsb  (Multi-gpu setting) Number of batches to run before synchronizing the weights of the different GPUs
      *  @param mem  Indicates the memory consumption of the model. One of "full_mem" (default), "mid_mem" or "low_mem".
      *  @return     The computer service itself.
    */
    compserv CS_GPU(const vector<int>& g, int lsb, const string& mem="full_mem");


    /**
      *  @brief Executes the code in the FPGA.
      *
      *  @param f  Vector of bools to set which FPGAs will be used (1=on, 0=off)
      *  @param lsb  (Multi-fpga setting) Number of batches to run before synchronizing the weights of the different FPGAs
      *  @return     The computer service itself.
    */
    compserv CS_FPGA(const vector<int> &f, int lsb=1);

    /**
      *  @brief Executes the code through the COMP Superscalar (COMPSs) framework.
      *
      *  @param filename  File with the setup specification
      *  @return     The computer service itself.
    */
    compserv CS_COMPSS(string filename);


    // Info and logs

    /**
      *  @brief  Save the training outputs of a model to a filename
      *
      *  @param net  Model to train
      *  @param fname  Name of the logfile
      *  @return     (void) Outputs log to the given file
    */
    void setlogfile(model net,string fname);
    /**
      *  @brief  Prints a summary representation of your model.
      *
      *  @param m  Model to print
      *  @return     (void) Prints the model
    */
    void summary(model m);
    /**
      *  @brief  Plots a representation of your model.
      *
      *  @param m  Model to plot
      *  @param fname  File where the plot will be saved
      *  @return     (void) Plots the model
    */
    void plot(model m, string fname, string mode="LR");

    // Serialization
    /**
      *  @brief  Load weights to reinstantiate your model.
      *
      *  @param m  Model
      *  @param fname  File where the model weights are saved
      *  @return     (void) Loads the weights
    */
    void load(model m, const string& fname, string format="bin");

    /**
      *  @brief  Save weights of a model.
      *
      *  @param m  Model
      *  @param fname  File where the model weights will be saved
      *  @return     (void) Saves the weights
    */
    void save(model m, const string& fname, string format="bin");

    // Optimizer
    /**
      *  @brief  Changes the learning rate and hyperparameters of the model optimizer.
      *
      *  @param net  Model
      *  @param p  Vector with the learning rate and hyperparameters of the model
      *  @return     (void) Changes model optimizer settings
    */
    void setlr(model net,vector<float>p);


    /**
      *  @brief Adadelta optimizer.
      *
      *  @details
      *   Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving window of gradient updates, instead of accumulating all past gradients. This way, Adadelta continues learning even when many updates have been done.
      *
      *  @see   https://arxiv.org/abs/1212.5701
      *
      *  @param lr  Learning rate
      *  @param rho  Smoothing constant
      *  @param epsilon   Term added to the denominator to improve numerical stability
      *  @param weight_decay   Weight decay (L2 penalty)
      *  @return    Adadelta optimizer
    */
    optimizer adadelta(float lr, float rho, float epsilon, float weight_decay); //Todo: Implement


    /**
      *  @brief Adam optimizer.
      *  @details Default parameters follow those provided in the original paper (See section).
      *  @see   https://arxiv.org/abs/1412.6980v8
      *
      *  @param lr  Learning rate
      *  @param beta_1  Coefficients used for computing running averages of gradient and its square
      *  @param beta_2  Coefficients used for computing running averages of gradient and its square
      *  @param epsilon   Term added to the denominator to improve numerical stability
      *  @param weight_decay   Weight decay (L2 penalty)
      *  @param amsgrad   Whether to apply the AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and Beyond".
      *  @return     Adam optimizer
    */
    optimizer adam(float lr=0.01, float beta_1=0.9, float beta_2=0.999, float epsilon=0.000001, float weight_decay=0,bool amsgrad=false); //Todo: Implement


    /**
      *  @brief Adagrad optimizer.
      *
      *  @details
      *   Adagrad is an optimizer with parameter-specific learning rates, which are adapted relative to how frequently a parameter gets updated during training. The more updates a parameter receives, the smaller the learning rate.
      *
      *  @see   http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
      *
      *  @param lr  Learning rate
      *  @param rho  Smoothing constant
      *  @param epsilon   Term added to the denominator to improve numerical stability
      *  @param weight_decay   Weight decay (L2 penalty)
      *  @return    Adagrad optimizer
    */
    optimizer adagrad(float lr, float epsilon, float weight_decay); //Todo: Implement

    /**
      *  @brief Adamax optimizer.
      *  @details
      *   It is a variant of Adam based on the infinity norm. Default parameters follow those provided in the See section.
      *  @see   https://arxiv.org/abs/1412.6980v8
      *
      *  @param lr  Learning rate
      *  @param beta_1  Coefficients used for computing running averages of gradient and its square
      *  @param beta_2  Coefficients used for computing running averages of gradient and its square
      *  @param epsilon   Term added to the denominator to improve numerical stability
      *  @param weight_decay   Weight decay (L2 penalty)
      *  @return     Adamax optimizer
    */
    optimizer adamax(float lr, float beta_1, float beta_2, float epsilon, float weight_decay); //Todo: Implement


    /**
      *  @brief Nadam optimizer.
      *  @details
      *   It is a variant of Adam based on the infinity norm. Default parameters follow those provided in the See section.
      *  @see   https://arxiv.org/abs/1412.6980v8
      *
      *  @param lr  Learning rate
      *  @param beta_1  Coefficients used for computing running averages of gradient and its square
      *  @param beta_2  Coefficients used for computing running averages of gradient and its square
      *  @param epsilon   Term added to the denominator to improve numerical stability
      *  @param schedule_decay   Weight decay (L2 penalty)
      *  @return     Nadam optimizer
    */
    optimizer nadam(float lr, float beta_1, float beta_2, float epsilon, float schedule_decay); //Todo: Implement


    /**
      *  @brief RMSProp optimizer.
      *
      *  @details
      *   It is recommended to leave the parameters of this optimizer at their default values (except the learning rate, which can be freely tuned).
      *
      *   @see  http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
      *
      *  @param lr  Learning rate
      *  @param rho  Smoothing constant
      *  @param epsilon   Term added to the denominator to improve numerical stability
      *  @param weight_decay   Weight decay (L2 penalty)
      *  @return     RMSProp optimizer
    */
    optimizer rmsprop(float lr=0.01, float rho=0.9, float epsilon=0.00001, float weight_decay=0.0); //Todo: Implement


    /**
      *  @brief Stochastic gradient descent optimizer.
      *
      *  @details
      *   Includes support for momentum, learning rate decay, and Nesterov momentum
      *
      *  @param lr  Learning rate
      *  @param momentum  Momentum factor
      *  @param weight_decay   Value to apply to the activation function
      *  @param nesterov   Boolean. Whether to apply Nesterov momentum
      *  @return     Stochastic gradient descent optimizer
    */
    optimizer sgd(float lr = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f, bool nesterov = false);

    // Training and Evaluation
    // Coarse methods
    /**
      *  @brief Trains the model for a fixed number of epochs (iterations on a dataset).
      *
      *  @param m  Model to train
      *  @param in  Input data (features)
      *  @param out  Output data (labels)
      *  @param batch  Number of samples per gradient update
      *  @param epochs  Number of epochs to train the model. An epoch is an iteration over the entire data provided
      *  @return     (void) Trains the model
    */
    void fit(model m, const vector<Tensor *> &in, const vector<Tensor *> &out, int batch, int epochs);
    /**
      *  @brief Returns the loss value & metrics values for the model in test mode.
      *
      *  @param m  Model to train
      *  @param in  Input data (features)
      *  @param out  Output data (labels)
      *  @param bs  Batch size (size [100])
      *  @return     (void) Evaluates the model
    */
    void evaluate(model m, const vector<Tensor *> &in, const vector<Tensor *> &out, int bs=-1);

    /**
      *  @brief Performs a prediction with input data
      *
      *  @param m  Model
      *  @param in  Input data (features)
      *  @return    vector of output tensors.
    */
    vector<Tensor *>  predict(model m, const vector<Tensor *> &in);


    // Finer methods

    /**
      *  @brief Generates a random sequence of indices for a batch
      *
      *  @param batch_size  Length of the random sequence to generate
      *  @param num_samples  Number of samples available, i.e. maximum value to include in the random sequence + 1
      *  @return    Vector of integers
    */
    vector<int> random_indices(int batch_size, int num_samples);

    /**
      *  @brief Train the model using the samples of the input vector that are on the selected indices vector
      *
      *  @param net Net to train
      *  @param in Vector of samples
      *  @param out Vector of labels or expected output
      *  @param indices Vector of indices of the samples to train
      *  @return    (void)
    */
    void train_batch(model net, vector<Tensor *> in, vector<Tensor *> out, vector<int> indices);

    /**
      *  @brief Evaluate the model using the samples of the input vector that are on the selected indices vector
      *
      *  @param net Net to evaluate
      *  @param in Vector of samples
      *  @param out Vector of labels or expected output
      *  @param indices Vector of indices of the samples to evaluate
      *  @return    (void)
    */
    void eval_batch(model net, vector<Tensor *> in, vector<Tensor *> out, vector<int> indices);

    /**
      *  @brief Loads the next batch of random samples from the input vector to the output vector
      *
      *  @param in Vector from where the samples of the next batch should be chosen from
      *  @param out Vector where the samples of the next batch should be stored
      *  @return    (void)
    */
    void next_batch(vector<Tensor *> in,vector<Tensor *> out);

    /**
      *  @brief Train the model using the samples of the input vector
      *
      *  @param net Net to train
      *  @param in Vector of samples
      *  @param out Vector of labels or expected output
      *  @return    (void)
    */
    void train_batch(model net, vector<Tensor *> in, vector<Tensor *> out);

    /**
      *  @brief Evaluate the model using the samples of the input vector
      *
      *  @param net Net to evaluate
      *  @param in Vector of samples
      *  @param out Vector of labels or expected output
      *  @return    (void)
    */
    void eval_batch(model net, vector<Tensor *> in, vector<Tensor *> out);

    // Finest methods
    /**
      *  @brief Set model mode.
      *
      *  @param net  Model
      *  @param mode  Train 1, Test 0
      *  @return     (void)
    */
    void set_mode(model net, int mode);
    /**
      *  @brief Resets model loss.
      *
      *  @param m  Model
      *  @return     (void)
    */
    void reset_loss(model m);
    /**
      *  @brief Computes the gradient of the model through the forward graph using the layers in the provided vector.
      *
      *  @param m  Model
      *  @param in Vector of Layer pointers
      *  @return     (void)
    */
    vlayer forward(model m,vector<Layer *> in);
    /**
      *  @brief Computes the gradient of the model through the forward graph using the tensors in the provided vector.
      *
      *  @param m  Model
      *  @param in Vector of Tensor pointers
      *  @return     (void)
    */
    vlayer forward(model m,vector<Tensor *> in);
    /**
      *  @brief Computes the gradient of the model through the forward graph.
      *
      *  @param m  Model
      *  @return     (void)
    */
    vlayer forward(model m);
    /**
      *  @brief Computes the gradient of the model through the forward graph and resizes the batch size of the model to ``b``.
      *
      *  @param m  Model
      *  @param b New batch size
      *  @return     (void)
    */
    vlayer forward(model m,int b);
    /**
      *  @brief Set model gradients to zero.
      *
      *  @param m  Model
      *  @return     (void)
    */
    void zeroGrads(model m);
    /**
      *  @brief Computes the gradient by passing its argument (1x1 unit tensor by default) through the backward graph.
      *
      *  @param m  Model
      *  @param target  Targets
      *  @return     (void)
    */
    void backward(model m,vector<Tensor *> target);
    /**
      *  @brief Computes the gradient of the model through the backward graph.
      *
      *  @param net  Model
      *  @return     (void)
    */
    void backward(model net);
    /**
      *  @brief Computes the gradient of the model associated to the given loss object through the backward graph.
      *
      *  @param l  Loss
      *  @return     (void)
    */
    void backward(loss l);
    void optimize(loss l);
    void optimize(vector <loss> l);
    /**
      *  @brief Updates the weights of the model
      *
      *  @param m  Model
      *  @return     (void)
    */
    void update(model m);
    /**
      *  @brief Prints model loss at some batch.
      *
      *  @param m  Model
      *  @param batch  Batch number
      *  @return     (void)
    */
    void print_loss(model m, int batch);

    /**
      *  @brief Get model losses
      *
      *  @param m  Model
      *  @return vector<float>
    */
    vector<float> get_losses(model m);

    /**
      *  @brief Get model metrics
      *
      *  @param m  Model
      *  @return vector<float>
    */
    vector<float> get_metrics(model m);

    // model constraints
    /**
      *  @brief Model parameters values clipping.
      *
      *  @param m  Model
      *  @param min  Minimum value
      *  @param max   Maximum value
      *  @return     (void) Performs model clamp between min and max
    */
    void clamp(model m,float min,float max);

    // loss and metrics methods
    /**
      *  @brief Computes loss of the associated model
      *
      *  @param L  Loss
      *  @return (float) Computed loss
    */
    float compute_loss(loss L);
    /**
      *  @brief Computes loss of the associated model (same as ``compute_loss``)
      *
      *  @param L  Loss
      *  @return (float) Computed loss
    */
    float compute_metric(loss L);
    /**
      *  @brief Get Loss object by its name.
      *
      *  @param type  Loss name/type
      *  @return     Selected Loss
    */
    Loss* getLoss(string type);
    /**
      *  @brief Create new Loss.
      *
      *  @param f  Loss function
      *  @param in  Loss input
      *  @param name  Loss name
      *  @return     Created Loss
    */
    loss newloss(const std::function<Layer*(vector<Layer*>)>& f, vector<Layer*> in, string name);
    /**
      *  @brief Create new Loss.
      *
      *  @param f  Loss function
      *  @param in  Loss input
      *  @param name  Loss name
      *  @return     Created Loss
    */
    loss newloss(const std::function<Layer*(Layer*)>& f, Layer *in, string name);
    /**
      *  @brief Get Metric object by its name.
      *
      *  @param type  Metric name/type
      *  @return     Selected Metric
    */
    Metric* getMetric(string type);
    /**
      *  @brief Create new Loss.
      *
      *  @param f  Loss function
      *  @param in  Loss input
      *  @param name  Loss name
      *  @return     Created Loss
    */
    loss newmetric(const std::function<Layer*(vector<Layer*>)>& f, vector<Layer*> in, string name);
    /**
      *  @brief Create new Loss.
      *
      *  @param f  Loss function
      *  @param in  Loss input
      *  @param name  Loss name
      *  @return     Created Loss
    */
    loss newmetric(const std::function<Layer*(Layer*)>& f, Layer *in, string name);

    // graph connections
    /**
      *  @brief Sets a layer as detached, excluding it from the computation of the gradients.
      *
      *  @param l  Layer to detach
      *  @return   Detached Layer
    */
    layer detach(layer l);

    /**
      *  @brief Sets the provided layers as detached, excluding them from the computation of the gradients.
      *
      *  @param l  Layers to detach
      *  @return   Detached Layers
    */
    vlayer detach(vlayer l);

    /**
      * @brief Shows profile information.
    */
    void show_profile();


    ///////////////////////////////////////
    //  LAYERS
    ///////////////////////////////////////

    // Core Layers
    /**
      *  @brief Solves non-linear equation with Newton method.
      *
      *  @details
      *   Applies an activation function to the given layer
      *
      *  @see   https://en.wikipedia.org/wiki/Activation_function
      *
      *  @param parent  Parent layer
      *  @param activation Name of the activation function
      *  @param params   Vector of floats representing the different params of the activation function
      *  (Examples: softmax=>{axis}, elu=>{alpha}, selu=>{alpha, scale}, leaky_relu=>{alpha}, linear=>{alpha})
      *  @param name  Name of the layer
      *  @return     Activation layer
    */
    layer Activation(layer parent, string activation, vector<float> params={}, string name="");


    layer SoftmaxDeprecated(layer parent, string name="");

    /**
      *  @brief Applies a Jacobian Softmax activation function to the given layer.
      *
      *  @see   https://en.wikipedia.org/wiki/Softmax_function
      *
      *  @param parent  Parent layer
      *  @param axis  Dimension in which to operate. Default -1, which uses the last axis
      *  @param name  Name of the layer
      *  @return     Output of Softmax transformation
    */
    layer Softmax(layer parent, int axis=-1, string name= "");

    /**
      *  @brief Applies a Sigmoid activation function to the given layer.
      *
      *  @see   https://en.wikipedia.org/wiki/Sigmoid_function
      *
      *  @param parent  Parent layer
      *  @param name  Name of the layer
      *  @return     Output of Sigmoid activation
    */
    layer Sigmoid(layer parent, string name="");

    /**
      *  @brief Applies a HardSigmoid activation function to the given layer.
      *
      *  @param parent  Parent layer
      *  @param name  Name of the layer
      *  @return     Output of HardSigmoid activation
    */
    layer HardSigmoid(layer parent, string name="");

    /**
      *  @brief Applies a Rectified Linear Unit activation function to the given layer.
      *
      *  @see   https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
      *
      *  @param parent  Parent layer
      *  @param name  Name of the layer
      *  @return     Output of ReLu activation
    */
    layer ReLu(layer parent, string name="");

    /**
      *  @brief Applies the Thresholded version of a Rectified Linear Unit activation function to the given layer.
      *
      *  @param parent  Parent layer
      *  @param alpha  Threshold value
      *  @param name  Name of the layer
      *  @return     Output of Thresholded ReLu activation
    */
    layer ThresholdedReLu(layer parent, float alpha=1.0, string name="");

    /**
      *  @brief Applies the Leaky version of a Rectified Linear Unit activation function to the given layer.
      *
      *  @see   https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLUs
      *
      *  @param parent  Parent layer
      *  @param alpha  Negative slope coefficient
      *  @param name  Name of the layer
      *  @return     Output of Leaky ReLu activation
    */
    layer LeakyReLu(layer parent, float alpha=0.01, string name="");

    /**
      *  @brief Applies the Exponential Linear Unit activation function to the given layer.
      *
      *  @param parent  Parent layer
	    *  @param alpha ELu coefficient
      *  @param name  Name of the layer
      *  @return     Output of ELu activation
    */
    layer Elu(layer parent, float alpha=1.0, string name="");

    /**
      *  @brief Applies the Scaled Exponential Linear Unit activation function to the given layer.
      *
      *  @param parent  Parent layer
      *  @param name  Name of the layer
      *  @return     Output of Selu activation
    */
    layer Selu(layer parent, string name="");

    /**
    *  @brief Applies the Exponential (base e) activation function to the given layer.
    *
    *  @param parent  Parent layer
    *  @param name  Name of the layer
    *  @return     Output of Exponential activation
    */
    layer Exponential(layer parent, string name="");

    /**
    *  @brief Applies the Softplus activation function to the given layer.
    *
    *  @param parent  Parent layer
    *  @param name  Name of the layer
    *  @return     Output of Exponential activation
    */
    layer Softplus(layer parent, string name="");


    /**
    *  @brief Applies the Softsign activation function to the given layer.
    *
    *  @param parent  Parent layer
    *  @param name  Name of the layer
    *  @return     Output of Exponential activation
    */
    layer Softsign(layer parent, string name="");

    /**
      *  @brief Applies the Linear activation function to the given layer.
      *
      *  @param parent  Parent layer
	    *  @param alpha Linear coefficient
      *  @param name  Name of the layer
      *  @return     Output of Linear activation
    */
    layer Linear(layer parent, float alpha=1.0, string name="");

    /**
      *  @brief Applies the Hyperbolic tangent activation function to the given layer.
      *
      *  @see   https://en.wikipedia.org/wiki/Hyperbolic_function
      *
      *  @param parent  Parent layer
      *  @param name  Name of the layer
      *  @return     Output of hyperbolic activation
    */
    layer Tanh(layer parent, string name="");

    /**
      *  @brief Convolution layer.
      *
      *  @param parent  Parent layer
      *  @param filters  Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)
      *  @param kernel_size  Vector of 2 integers, specifying the height and width of the 2D convolution window
      *  @param strides  Vector of 2 integers, specifying the strides of the convolution along the height and width
      *  @param padding  One of "none", "valid" or "same"
      *  @param use_bias  Boolean, whether the layer uses a bias vector
      *  @param groups  Number of blocked connections from input channels to output channels
      *  @param dilation_rate  Vector of 2 integers, specifying the dilation rate to use for dilated convolution
      *  @param name  A name for the operation
      *  @return     Convolution layer
    */
    layer Conv(layer parent, int filters, const vector<int> &kernel_size,
               const vector<int> &strides = {1, 1}, string padding = "same", bool use_bias = true,
               int groups = 1, const vector<int> &dilation_rate = {1, 1}, string name = "");


    /**
   *  @brief 1D Convolution layer.
   *
   *  @param parent  Parent layer
   *  @param filters  Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)
   *  @param kernel_size  Vector of 1 integers, specifying the height and width of the 2D convolution window
   *  @param strides  Vector of 1 integers, specifying the strides of the convolution along the height and width
   *  @param padding  One of "none", "valid" or "same"
   *  @param use_bias  Boolean, whether the layer uses a bias vector
   *  @param groups  Number of blocked connections from input channels to output channels
   *  @param dilation_rate  Vector of 1 integers, specifying the dilation rate to use for dilated convolution
   *  @param name  A name for the operation
   *  @return     Convolution layer
 */
    layer Conv1D(layer parent, int filters,  vector<int> kernel_size,
                 vector<int> strides = {1}, string padding = "same", bool use_bias = true,
                 int groups = 1, const vector<int> dilation_rate = {1}, string name = "");


    /**
  *  @brief 2D Convolution layer.
  *
  *  @param parent  Parent layer
  *  @param filters  Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)
  *  @param kernel_size  Vector of 2 integers, specifying the height and width of the 2D convolution window
  *  @param strides  Vector of 2 integers, specifying the strides of the convolution along the height and width
  *  @param padding  One of "none", "valid" or "same"
  *  @param use_bias  Boolean, whether the layer uses a bias vector
  *  @param groups  Number of blocked connections from input channels to output channels
  *  @param dilation_rate  Vector of 2 integers, specifying the dilation rate to use for dilated convolution
  *  @param name  A name for the operation
  *  @return     Convolution layer
*/
    layer Conv2D(layer parent, int filters, const vector<int> &kernel_size,
               const vector<int> &strides = {1, 1}, string padding = "same", bool use_bias = true,
               int groups = 1, const vector<int> &dilation_rate = {1, 1}, string name = "");

    /**
    *  @brief 3D Convolution layer.
    *
    *  @param parent  Parent layer
    *  @param filters  Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)
    *  @param kernel_size  Vector of 3 integers, specifying the depth, height and width of the 3D convolution window
    *  @param strides  Vector of 3 integers, specifying the strides of the convolution along the depth, height and width
    *  @param padding  One of "none", "valid" or "same"
    *  @param use_bias  Boolean, whether the layer uses a bias vector
    *  @param groups  Number of blocked connections from input channels to output channels
    *  @param dilation_rate  Vector of 3 integers, specifying the dilation rate to use for dilated convolution
    *  @param name  A name for the operation
    *  @return     Convolution layer
    */
    layer Conv3D(layer parent, int filters, const vector<int> &kernel_size,
                 const vector<int> &strides = {1, 1, 1}, string padding = "same", bool use_bias = true,
                 int groups = 1, const vector<int> &dilation_rate = {1, 1, 1}, string name = "");

    /**
   *  @brief 2D Transposed convolution layer (sometimes called Deconvolution).
   *
   *  @details
   *   The need for transposed convolutions generally arises from the desire to use a transformation going in the opposite direction of a normal convolution, i.e., from something that has the shape of the output of some convolution to something that has the shape of its input while maintaining a connectivity pattern that is compatible with said convolution.
   *
   *  @param parent  Parent layer
   *  @param filters  The dimensionality of the output space (i.e. the number of output filters in the convolution)
   *  @param kernel_size  The height and width of the 2D convolution window
   *  @param padding  One of "valid" or "same"
   *  @param dilation_rate  The dilation rate to use for dilated convolution. Spacing between kernel elements
   *  @param strides  The strides of the convolution along the height and width
   *  @param use_bias  Boolean, whether the layer uses a bias vector
   *  @param name  A name for the operation
   *  @return     Output layer after upsampling operation
 */

  layer ConvT2D(layer parent, int filters, const vector<int> &kernel_size,
                 const vector<int> &strides = {1, 1}, string padding = "same", bool use_bias = true,
                 int groups = 1, const vector<int> &dilation_rate = {1, 1}, string name = "");

    /**
 *  @brief 3D Transposed convolution layer (sometimes called Deconvolution).
 *
 *  @details
 *   The need for transposed convolutions generally arises from the desire to use a transformation going in the opposite direction of a normal convolution, i.e., from something that has the shape of the output of some convolution to something that has the shape of its input while maintaining a connectivity pattern that is compatible with said convolution.
 *
 *  @param parent  Parent layer
 *  @param filters  The dimensionality of the output space (i.e. the number of output filters in the convolution)
 *  @param kernel_size  The depth, height and width of the 3D convolution window
 *  @param padding  One of "valid" or "same"
 *  @param dilation_rate  The dilation rate to use for dilated convolution. Spacing between kernel elements
 *  @param strides  The strides of the convolution along the depth, height and width
 *  @param use_bias  Boolean, whether the layer uses a bias vector
 *  @param name  A name for the operation
 *  @return     Output layer after upsampling operation
*/
    layer ConvT3D(layer parent, int filters, const vector<int> &kernel_size,
                  const vector<int> &strides = {1, 1, 1}, string padding = "same", bool use_bias = true,
                  int groups = 1, const vector<int> &dilation_rate = {1, 1, 1}, string name = "");

    /**
      *  @brief Pointwise 2D convolution
      *
      *  @param parent  Parent layer
      *  @param filters  Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)
      *  @param strides  Vector of 2 integers, specifying the strides of the convolution along the height and width
      *  @param use_bias  Boolean, whether the layer uses a bias vector
      *  @param groups  Number of blocked connections from input channels to output channels
      *  @param dilation_rate  Vector of 2 integers, specifying the dilation rate to use for dilated convolution
      *  @param name  A name for the operation
      *  @return     Convolution layer
    */
    layer PointwiseConv2D(layer parent, int filters, const vector<int> &strides = {1, 1}, bool use_bias = true,
               int groups = 1, const vector<int> &dilation_rate = {1, 1}, string name = "");

    // Legacy
    layer PointwiseConv(layer parent, int filters, const vector<int> &strides = {1, 1}, bool use_bias = true,
                        int groups = 1, const vector<int> &dilation_rate = {1, 1}, string name = "");

    /**
      *  @brief DepthwiseConv 2D convolution
      *
      *  @param parent  Parent layer
      *  @param kernel_size  The depth, height and width of the convolution window
      *  @param strides  Vector of 2 integers, specifying the strides of the convolution along the height and width
      *  @param use_bias  Boolean, whether the layer uses a bias vector
      *  @param dilation_rate  Vector of 2 integers, specifying the dilation rate to use for dilated convolution
      *  @param name  A name for the operation
      *  @return     Convolution layer
    */
    layer DepthwiseConv2D(layer parent, const vector<int> &kernel_size, const vector<int> &strides = {1, 1}, string padding = "same",
                          bool use_bias = true, const vector<int> &dilation_rate = {1, 1}, string name = "");

    /**
      *  @brief Regular densely-connected NN layer.
      *
      *  @param parent  Parent layer
      *  @param ndim  Positive integer, dimensionality of the output space
      *  @param use_bias  Boolean, whether the layer uses a bias vector.
      *  @param name  A name for the operation
      *  @return     Densely-connected NN layer
    */
    layer Dense(layer parent, int ndim, bool use_bias = true,  string name = "");

    /**
      *  @brief Applies Dropout to a layer.
      *
      *  @details
      *   Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
      *
      *  @param parent  Parent layer
      *  @param rate  Between 0 and 1. Fraction of the input units to drop
      *  @param iw  perform weighting in inference (boolean, true)
      *  @param name  A name for the operation
      *  @return     Layer with Dropout
    */
    layer Dropout(layer parent, float rate, bool iw=true, string name = "");

    /**
      *  @brief Used to initialize an input to a model.
      *
      *  @param shape  A shape vector (integer), not including the batch size. For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors
      *  @param name  A name for the operation
      *  @return     Input layer
    */
    layer Input(const vector<int> &shape, string name = "");

    /**
      *  @brief 2D Upsampling layer.
      *
      *  @details
      *   Identical to the ``scale`` transformation, it is an alias of the Resize layer.
      *
      *  @param parent  Parent layer
      *  @param size  Vector of 2 integers. The upsampling factors for rows and columns
      *  @param interpolation (Deprecated) A string, only "nearest" is valid
      *  @param name  A name for the operation
      *  @return     Output layer after upsampling operation
    */
    layer UpSampling2D(layer parent, const vector<int> &size, string interpolation = "nearest", string name = "");
    layer UpSampling(layer parent, const vector<int> &size, string interpolation = "nearest", string name = "");

    /**
      *  @brief 3D Upsampling layer. Similar to Resize but for 3D images
      *
      *  @param parent  Parent layer
      *  @param new_shape  Vector with layer/images desired new shape
      *  @param reshape  If True, the output shape will be new_shape (classical scale; recommended). If False, the output shape will be the input shape (scale<100%: scale + padding; scale >100%: crop + scale)
      *  @param da_mode  One of "nearest", "constant", (ToDo: "mirror", "reflect", "wrap", "original")
      *  @param constant  Fill value for area outside the resized image, it is used for all channels respectively
      *  @param coordinate_transformation_mode  This attribute describes how to transform the coordinate in the resized tensor to the coordinate in the original tensor.
      *  @return     Output of scale transformation
    */
    layer UpSampling3D(layer parent, vector<int> new_shape, bool reshape=true, string da_mode="constant", float constant=0.0f, string coordinate_transformation_mode="asymmetric", string name="");


    /**
      *  @brief Resize the input image to the given size. `[height, width]`. Same as the Scale layer, but with the backward operation supported
      *
      *  @param parent  Parent layer
      *  @param new_shape  Vector with layer/images desired new shape
      *  @param reshape  If True, the output shape will be new_shape (classical scale; recommended). If False, the output shape will be the input shape (scale<100%: scale + padding; scale >100%: crop + scale)
      *  @param da_mode  One of "nearest", "constant", (ToDo: "mirror", "reflect", "wrap", "original")
      *  @param constant  Fill value for area outside the resized image, it is used for all channels respectively
      *  @param coordinate_transformation_mode  This attribute describes how to transform the coordinate in the resized tensor to the coordinate in the original tensor.
      *  @return     Output of scale transformation
    */
    layer Resize(layer parent, vector<int> new_shape, bool reshape=true, string da_mode="constant", float constant=0.0f, string coordinate_transformation_mode="asymmetric", string name="");

    /**
      *  @brief Reshapes an output to a certain shape.
      *
      *  @param parent  Parent layer
      *  @param shape  Target shape. Vector of integers. Does not include the batch axis
      *  @param name  A name for the operation
      *  @return     Output of reshape operation
    */
    layer Reshape(layer parent, const vector<int> &shape, string name = "");

    /**
      *  @brief Flattens the input. Does not affect the batch size.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     Output of reshape operation
    */
    layer Flatten(layer parent, string name = "");

    /**
      *  @brief Repeats the elements of a tensor (layer's output) along the specified dimension.
      *
      *  @param parent  Parent layer
      *  @param repeats  The number of repetitions for the specified dimension ("int" or "vector of ints")
      *  @param axis  The axis along which to repeat values. (Batch is ignored)
      *  @return     Output of repeat operation
    */
    layer Repeat(layer parent, const vector<unsigned int>& repeats, unsigned int axis, string name="");
    layer Repeat(layer parent, unsigned int repeats, unsigned int axis, string name="");


    /**
      *  @brief Dimension of size one is removed at the specified position. (Batch dimension is ignored)
      *
      *  @param parent  Parent layer
      *  @param  axis if given, the input will be squeezed only in this dimension. Else (-1), squeezes all
      *  @param name  A name for the operation
      *  @return     Output of reshape operation
    */
    layer Squeeze(layer parent, int axis=-1, string name = "");

        /**
      *  @brief Dimension of size one is inserted at the specified position. (Batch dimension is ignored)
      *
      *  @param parent  Parent layer
      *  @param  axis if given, the input will be unsqueezed only in this dimension
      *  @param name  A name for the operation
      *  @return     Output of reshape operation
    */
    layer Unsqueeze(layer parent, int axis=0, string name = "");

    /**
      *  @brief Turns positive integers (indexes) into dense vectors of fixed size. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
      *
      *  @param parent Parent layer
      *  @param vocsize Size of the vocabulary, i.e. maximum integer index + 1
      *  @param output_dim  Dimension of the dense embedding
      *  @param length (1) Length of the sequence, to connect to Dense Layers no Recurrent
      *  @param name  A name for the operation
      *  @return The embedded input
    */
    layer Embedding(layer parent, int vocsize, int length, int output_dim,  bool mask_zeros=false, string name = ""); //Todo: Implement

    /**
      *  @brief Transposes a Layer.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     Output of transpose operation
    */
    layer Transpose(layer parent, string name = "");

    /**
      *  @brief Given a tensor (constant), this layer outputs the same tensor but repeat across the batch.
      *
      *  @param Tensor  Raw tensor
      *  @param name  A name for the operation
      *  @return Raw repeated for each batch
    */
    layer ConstOfTensor(Tensor* t, string name = "");

    /**
      *  @brief Return elements chosen from x or y depending on condition.
      *
      *  @param parent1  Parent layer
      *  @param parent2  Parent layer
      *  @param condition  Condition layer. Where True, selects parent1; where False, selects parent2. (Bool => Float of 0.0s and 1.0s)
      *  @param name  A name for the operation
      *  @return Raw repeated for each batch
    */
    layer Where(layer parent1, layer parent2, layer condition, string name = "");

    // Transformation Layers
    /**
      *  @brief Affine transformation of the image keeping center invariant: rotate+translate+scale+shear.
      *
      *  @param parent  Parent layer
      *  @param angle  Angle factor
      *  @param translate  Translate factor
      *  @param scale  Scaling factor
      *  @param shear  Shear factor
      *  @param name  A name for the operation
      *  @return     Output of affine transformation
    */
    layer Affine(layer parent, float angle=0, float translate=0, float scale=0, float shear=0, string name="");  // TODO: Implement

    /**
      *  @brief Crops the given image at `[(top, left), (bottom, right)]`.
      *
      *  @param parent  Parent layer
      *  @param from_coords  Vector (top, left) coordinates
      *  @param to_coords  Vector (bottom, right) coordinates
      *  @param reshape  If True, the output shape will be new_shape (classical scale; recommended). If False, the output shape will be the input shape (scale<100%: scale + padding; scale >100%: crop + scale)
      *  @param constant  Erasing value
      *  @param name  A name for the operation
      *  @return     Output of crop transformation
    */
    layer Crop(layer parent, vector<int> from_coords, vector<int> to_coords, bool reshape=true, float constant=0.0f, string name="");

    /**
      *  @brief Crops the given image at the center with size `(width, height)`.
      *
      *  @param parent  Parent layer
      *  @param size  Vector (height, width) size
      *  @param reshape  If True, the output shape will be new_shape (classical scale; recommended). If False, the output shape will be the input shape (scale<100%: scale + padding; scale >100%: crop + scale)
      *  @param constant  Erasing value
      *  @param name  A name for the operation
      *  @return     Output of center crop transformation
    */
    layer CenteredCrop(layer parent, vector<int> size, bool reshape=true, float constant=0.0f, string name="");

    /**
      *  @brief Randomly change the brightness, contrast and saturation of an image.
      *
      *  @param parent  Parent layer
      *  @param brightness  How much to jitter brightness. brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. Should be non negative numbers
      *  @param contrast  How much to jitter contrast. contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. Should be non negative numbers
      *  @param saturation  How much to jitter saturation. saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. Should be non negative numbers
      *  @param hue  How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue] or the given [min, max]. Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5
      *  @param name  A name for the operation
      *  @return     Output of color jitter transformation
    */
    layer ColorJitter(layer parent, float brightness=0, float contrast=0, float saturation=0, float hue=0, string name="");  // TODO: Implement

    /**
      *  @brief Crop the given image at `[(top, left), (bottom, right)]` and scale it to the parent size.
      *
      *  @param parent  Parent layer
      *  @param from_coords  Vector (top, left) coordinates
      *  @param to_coords  Vector (bottom, right) coordinates
      *  @param da_mode  One of "nearest", "constant", (ToDo: "mirror", "reflect", "wrap", "original")
      *  @param constant  Fill value for area outside the rotated image, it is used for all channels respectively
      *  @param name  A name for the operation
      *  @return     Output of crop scale transformation
    */
    layer CropScale(layer parent, vector<int> from_coords, vector<int> to_coords, string da_mode="constant", float constant=0.0f, string name="");

    /**
      *  @brief Selects a rectangle region in an image at `[(top, left), (bottom, right)]` and erases its pixels using a constant value.
      *
      *  @param parent  Parent layer
      *  @param from_coords  Vector (top, left) coordinates
      *  @param to_coords  Vector (bottom, right) coordinates
      *  @param constant  Erasing value
      *  @param name  A name for the operation
      *  @return     Output of cutout transformation
    */
    layer Cutout(layer parent, vector<int> from_coords, vector<int> to_coords, float constant=0.0f, string name="");

    /**
      *  @brief Flip the given image at `axis=n`.
      *
      *  @param parent  Parent layer
      *  @param axis  Flip axis
      *  @param name  A name for the operation
      *  @return     Output of flip transformation
    */
    layer Flip(layer parent, int axis=0, string name="");

    /**
      *  @brief Convert image to grayscale.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     Output of grayscale transformation
    */
    layer Grayscale(layer parent,  string name="");  // TODO: Implement

    /**
      *  @brief Horizontally flip the given image.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     Output of horizontal flip transformation
    */
    layer HorizontalFlip(layer parent, string name="");

    /**
      *  @brief Pad the given image on all sides with the given `pad` value.
      *
      *  @param parent  Parent layer
      *  @param padding  Padding on each border (top-bottom, left-right) or (top, right, bottom, left)
      *  @param constant  pads with a constant value
      *  @param name  A name for the operation
      *  @return     Padded image
    */
    layer Pad(layer parent, vector<int> padding, float constant=0.0f, string name="");

    /**
      *  @brief Rotate the image by angle.
      *
      *  @param parent  Parent layer
      *  @param angle  In degrees counter clockwise order
      *  @param offset_center  Optional center of rotation. Default is the center of the image
      *  @param da_mode  One of "nearest", "constant", (ToDo: "mirror", "reflect", "wrap", "original")
      *  @param constant  Fill value for area outside the rotated image, it is used for all channels respectively
      *  @return     Output of rotate transformation
    */
    layer Rotate(layer parent, float angle, vector<int> offset_center={0, 0}, string da_mode="original", float constant=0.0f, string name="");

    /**
      *  @brief Resize the input image to the given size. `[height, width]`.
      *
      *  @param parent  Parent layer
      *  @param new_shape  Vector with layer/images desired new shape
      *  @param reshape  If True, the output shape will be new_shape (classical scale; recommended). If False, the output shape will be the input shape (scale<100%: scale + padding; scale >100%: crop + scale)
      *  @param da_mode  One of "nearest", "constant", (ToDo: "mirror", "reflect", "wrap", "original")
      *  @param constant  Fill value for area outside the resized image, it is used for all channels respectively
      *  @param coordinate_transformation_mode  This attribute describes how to transform the coordinate in the resized tensor to the coordinate in the original tensor.
      *  @return     Output of scale transformation
    */
    layer Scale(layer parent, vector<int> new_shape, bool reshape=true, string da_mode="constant", float constant=0.0f, string coordinate_transformation_mode="asymmetric", string name="");

    /**
      *  @brief Shift the input image `[a, b]`.
      *
      *  @param parent  Parent layer
      *  @param shift  Vector of maximum absolute fraction for horizontal and vertical translations
      *  @param da_mode  One of "nearest", "constant", (ToDo: "mirror", "reflect", "wrap", "original")
      *  @param constant  Fill value for area outside the resized image, it is used for all channels respectively
      *  @return     Output of scale transformation
    */
    layer Shift(layer parent, vector<int> shift, string da_mode="nearest", float constant=0.0f, string name="");

    /**
      *  @brief Vertically flip the given image.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     Output of vertical flip transformation
    */
    layer VerticalFlip(layer parent, string name="");

    /**
      *  @brief Normalize an image with mean and standard deviation.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     Output of normalize transformation
    */
    layer Normalize(layer parent, string name="");  // TODO: Implement


    // Data augmentation Layers
    /**
      *  @brief Random affine transformation of the image keeping center invariant: rotate+translate+scale+shear.
      *
      *  @param parent  Parent layer
      *  @param angle  Angle factor range
      *  @param translate  Translate factor range
      *  @param scale  Scaling factor range
      *  @param shear  Shear factor range
      *  @param name  A name for the operation
      *  @return     Output of affine transformation
    */
    layer RandomAffine(layer parent, vector<float> angle, vector<float> translate, vector<float> scale, vector<float> shear, string name="");  // TODO: Implement

    /**
      *  @brief Crop the given image at a random location with size `[height, width]`.
      *
      *  @param parent  Parent layer
      *  @param new_shape  Vector (height, width) size
      *  @param name  A name for the operation
      *  @return     Output of random crop transformation
    */
    layer RandomCrop(layer parent, vector<int> new_shape, string name= "");

    /**
     *  @brief Crop the given image randomly by the size in a range `[a, b]` by and scale it to the parent size.
     *
     *  @param parent  Parent layer
     *  @param factor  Factor Range for random crop
     *  @param da_mode  One of "nearest", "constant", (ToDo: "mirror", "reflect", "wrap", "original")
     *  @param name  A name for the operation
     *  @return     Output of random crop scale transformation
   */
    layer RandomCropScale(layer parent, vector<float> factor, string da_mode= "nearest", string name= "");

    /**
      *  @brief Randomly selects a rectangle region in an image and erases its pixels. The random region is defined by the range `[(min_x, max_x), (min_y, max_y)]`, where these are relative values.
      *
      *  @param parent  Parent layer
      *  @param factor_x  Vector of factor fraction for horizontal size
      *  @param factor_y  Vector of factor fraction for vertical size
      *  @param constant  Erasing value
      *  @param name  A name for the operation
      *  @return     Output of random cutout transformation
    */
    layer RandomCutout(layer parent, vector<float> factor_x, vector<float> factor_y, float constant= 0.0f, string name= "");

    /**
      *  @brief Flip the given image at `axis=n` randomly with a given probability.
      *
      *  @param parent  Parent layer
      *  @param axis  Flip axis
      *  @param name  A name for the operation
      *  @return     Output of random flip transformation
    */
    layer RandomFlip(layer parent, int axis, string name= "");

    /**
      *  @brief Converts the given image to grayscale a given probability.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     Output of random horizontal flip transformation
    */
    layer RandomGrayscale(layer parent, string name= "");

    /**
      *  @brief Horizontally flip the given image randomly with a given probability.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     Output of random horizontal flip transformation
    */
    layer RandomHorizontalFlip(layer parent, string name= "");

    /**
      *  @brief Rotate the image randomly by an angle defined in a range `[a, b]`.
      *
      *  @param parent  Parent layer
      *  @param factor  Range In degrees counter clockwise order
      *  @param offset_center  Optional center of rotation. Default is the center of the image
      *  @param da_mode  One of "original"
      *  @param constant  Fill value for area outside the rotated image, it is used for all channels respectively.
      *  @return     Output of rotate transformation
    */
    layer RandomRotation(layer parent, vector<float> factor, vector<int> offset_center= {0, 0}, string da_mode= "original", float constant= 0.0f, string name= "");

    /**
      *  @brief Resize the input image randomly by the size in a range `[a, b]`.
      *
      *  @param parent  Parent layer
      *  @param factor  Vector of factor size range new shape
      *  @param da_mode  One of "nearest"
      *  @param constant  Fill value for area outside the resized image, it is used for all channels respectively.
      *  @param coordinate_transformation_mode  This attribute describes how to transform the coordinate in the resized tensor to the coordinate in the original tensor.
      *  @return     Output of scale transformation
    */
    layer RandomScale(layer parent, vector<float> factor, string da_mode= "nearest", float constant= 0.0f, string coordinate_transformation_mode="asymmetric", string name= "");

    /**
      *  @brief Shift the input image randomly in range `[a, b]`.
      *
      *  @param parent  Parent layer
      *  @param factor_x  Vector of factor fraction for horizontal translations
      *  @param factor_y  Vector of factor fraction for vertical translations
      *  @param da_mode  One of "nearest", "constant", (ToDo: "mirror", "reflect", "wrap", "original")
      *  @param constant  Fill value for area outside the resized image, it is used for all channels respectively.
      *  @return     Output of scale transformation
    */
    layer RandomShift(layer parent, vector<float> factor_x, vector<float> factor_y, string da_mode= "nearest", float constant= 0.0f, string name= "");

    /**
      *  @brief Vertically flip the given image randomly with a given probability.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     Output of random vertical flip transformation
    */
    layer RandomVerticalFlip(layer parent, string name= "");

    // Merge Layers
    /**
      *  @brief Layer that adds a list of layer inputs.
      *
      *  @details
      *   It takes as input a list of layers, all of the same shape, and returns a single tensor (also of the same shape).
      *
      *  @param layers  List of layers
      *  @param name  A name for the operation
      *  @return     Output of add operation with all input layers
    */
    layer Add(const vector<layer> &layers, string name = "");

    /**
      *  @brief Layer that averages a list of layer inputs.
      *
      *  @details
      *   It takes a list of layers as input, all of the same shape, and returns a single tensor (also of the same shape).
      *
      *  @param layers  List of layers
      *  @param name  A name for the operation
      *  @return     Output of average operation with all input layers
    */
    layer Average(const vector<layer> &layers, string name = ""); //Todo: Implement

    /**
      *  @brief Layer that concatenates a list of inputs.
      *
      *  @details
      *   It takes a list of layers as input and returns a single tensor that is the concatenation of all the input layers.
      *
      *  @param layers  List of layers
      *  @param axis  Axis along which to concatenate (batch is ignored; "-1" selects last axis)
      *  @param name  A name for the operation
      *  @return     Output of concatenation operation with all input layers
    */
    layer Concat(const vector<layer> &layers, unsigned int axis=0, string name = "");

    /**
     *  @brief Multiplication of matrices.
     *
     *  @details It takes a list of layers as input, all of the same shape, and returns a single tensor (also of the same shape).
     *
     *  @param layers List of layers
     *  @param name A name for the operation
     *
     *  @return Output of MatMul operation
     *
    */
    layer MatMul(const vector<layer> &layers, string name = "");

    /**
      *  @brief Layer that computes the maximum (element-wise) of a list of inputs.
      *
      *  @details
      *   It takes a list of tensors as input, all of the same shape, and returns a single tensor (also of the same shape).
      *
      *  @param layers  List of layers
      *  @param name  A name for the operation
      *  @return     Output of Maximum operation with all input layers
    */
    layer Maximum(const vector<layer> &layers, string name = "");


    /**
      *  @brief Layer that computes the minimum (element-wise) of a list of inputs.
      *
      *  @details
      *   It takes a list of tensors as input, all of the same shape, and returns a single tensor (also of the same shape).
      *
      *  @param layers  List of layers
      *  @param name  A name for the operation
      *  @return     Output of Minimum operation with all input layers
    */
    layer Minimum(const vector<layer> &layers, string name = "");


    /**
      *  @brief Layer that subtracts two inputs.
      *
      *  @details
      *   It takes a list of tensors of size 2 as input, both of the same shape, and returns a single tensor (inputs[0] - inputs[1]), also of the same shape.
      *
      *  @param layers  List of layers
      *  @param name  A name for the operation
      *  @return     Output of Substract operation with all input layers
    */
    layer Subtract(const vector<layer> &layers, string name = "");


    // Noise Layers
    /**
      *  @brief Apply additive zero-centered Gaussian noise.
      *
      *  @details
      *   This is useful to mitigate overfitting (you could see it as a form of random data augmentation).
      *   Gaussian Noise (GS) is a natural choice as corruption process for real valued inputs.
      *   As it is a regularization layer, it is only active at training time.
      *
      *  @param parent  Parent layer
      *  @param stddev  Standard deviation of the noise distribution
      *  @param name  A name for the operation
      *  @return     The parent after applying the GaussianNoise layer
    */
    layer GaussianNoise(layer parent, float stddev, string name = "");

    // Normalization
    /**
      *  @brief Batch normalization layer.
      *
      *  @details
      *   Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
      *
      *  @see   https://arxiv.org/abs/1502.03167
      *
      *  @param parent  Parent layer
      *  @param momentum  Momentum for the moving mean and the moving variance
      *  @param epsilon  Small float added to variance to avoid dividing by zero
      *  @param affine  If True, this module has learnable affine parameters
      *  @param name  A name for the operation
      *  @return     Parent layer after the normalization
    */
    layer BatchNormalization(layer parent, float momentum = 0.99f, float epsilon = 0.001f, bool affine = true,string name = "");
    layer BatchNormalization(layer parent, bool affine, float momentum = 0.99f, float epsilon = 0.001f, string name = "");

    /**
      *  @brief Layer normalization layer.
      *
      *  @details
      *   Applies Layer Normalization over an input.
      *
      *  @see   https://arxiv.org/abs/1607.06450
      *
      *  @param parent  Parent layer
      *  @param epsilon  Value added to the denominator for numerical stability
      *  @param affine  If True, this module has learnable affine parameters
      *  @param name  A name for the operation
      *  @return     Parent layer after the normalization
    */
    layer LayerNormalization(layer parent, float epsilon = 0.00001f, bool affine=true, string name = "");
    layer LayerNormalization(layer parent, bool affine,float epsilon = 0.00001f,  string name = "");

    /**
      *  @brief Group normalization layer.
      *
      *  @details
      *   Divides the channels into groups and computes within each group the mean and variance for normalization. The computation is independent of batch sizes.
      *
      *  @see   https://arxiv.org/abs/1803.08494
      *
      *  @param parent  Parent layer
      *  @param groups  Number of groups in which the channels will be divided
      *  @param epsilon  Value added to the denominator for numerical stability
      *  @param affine  If True, this module has learnable affine parameters
      *  @param name  A name for the operation
      *  @return     Parent layer after the normalization
    */
    layer GroupNormalization(layer parent, int groups, float epsilon = 0.001f, bool affine=true, string name = "");


    layer Norm(layer parent, float epsilon = 0.001f, string name = "");
    layer NormMax(layer parent, float epsilon = 0.001f, string name = "");
    layer NormMinMax(layer parent, float epsilon = 0.001f, string name = "");


    //  Operator Layers

    /**
      *  @brief Computes the element-wise absolute value of the given input tensor.
      *
      *  @param l  Parent layer
      *  @return     Parent layer `l` after computing the element-wise absolute value
    */
    layer Abs(layer l);

    /**
      *  @brief Layer that computes the difference of two layers.
      *
      *  @param l1  Layer
      *  @param l2  Layer
      *  @return Difference between `l1` and `l2`
    */
    layer Sub(layer l1, layer l2);

    /**
      *  @brief Layer that computes the difference of a layer and a float number.
      *
      *  @param l1  Parent layer
      *  @param k  Number
      *  @return     Parent layer `l1` after computing its difference with `k`
    */
    layer Sub(layer l1, float k);
    layer Sub(float k, layer l1);

    // Deprecate aliases
    layer Diff(layer l1, layer l2);
    layer Diff(layer l1, float k);
    layer Diff(float k, layer l1);

    /**
      *  @brief Layer that computes the element-wise division of two layers.
      *
      *  @param l1  Layer
      *  @param l2  Layer
      *  @return     Element-wise division of `l1` and `l2`
    */
    layer Div(layer l1, layer l2);
    /**
      *  @brief Layer that computes the division of a layer by a float number.
      *
      *  @param l1  Parent layer
      *  @param k  Number
      *  @return     Parent layer `l1` after dividing it by `k`
    */
    layer Div(layer l1, float k);
    layer Div(float k, layer l1);

    /**
      *  @brief Returns a new tensor with the exponential of the elements of the input tensor.
      *
      *  @param l  Parent layer
      *  @return     The exponential of `l`
    */
    layer Exp(layer l);


    /**
      *  @brief Layer that computes the logarithm of a layer.
      *
      *  @param l  Parent layer
      *  @return     Parent layer `l` after computing its logarithm
    */
    layer Log(layer l);

    /**
      *  @brief Layer that computes the logarithm to base 2 of a layer.
      *
      *  @param l  Parent layer
      *  @return     Parent layer `l` after computing its logarithm to base 2
    */
    layer Log2(layer l);

    /**
      *  @brief Layer that computes the logarithm to base 10 of a layer.
      *
      *  @param l  Parent layer
      *  @return     Parent layer `l` after computing its logarithm to base 10
    */
    layer Log10(layer l);

    /**
      *  @brief Clamps all elements in input into the range [ min, max ].
      *
      *  @param parent  Parent layer
      *  @param min  Lower-bound of the range to be clamped to
      *  @param max  Upper-bound of the range to be clamped to
      *  @param name  A name for the operation
      *  @return     Output of reshape operation
    */
    layer Clamp(layer parent, float min, float max, string name = "");
    layer Clip(layer parent, float min, float max, string name = "");

    /**
      *  @brief  Layer that computes the element-wise multiplication of two layers.
      *
      *  @param l1  Layer
      *  @param l2  Layer
      *  @return     Result of the element-wise multiplication of `l1` and `l2`
    */
    layer Mult(layer l1, layer l2);
    /**
      *  @brief Layer that computes the multiplication of a float number and a layer.
      *
      *  @param l1  Parent layer
      *  @param k  Number
      *  @return     Parent layer `l1` after multiplying its elements by `k`
    */
    layer Mult(layer l1, float k);
    layer Mult(float k,layer l1);

    /**
      *  @brief Layer that computes the power of a layer raised to a float number.
      *
      *  @param l1  Parent layer
      *  @param k  Number
      *  @return     Parent layer `l1` after computing its power raised to `k`
    */
    layer Pow(layer l1, float k);

    /**
      *  @brief  Layer that computes the square root of a layer.
      *
      *  @param l  Parent layer
      *  @return     Result of the square root of `l`
    */
    layer Sqrt(layer l);

    /**
      *  @brief Layer that computes the sum of two layers.
      *
      *  @param l1  Layer
      *  @param l2  Layer
      *  @return     The result after computing the sum of layers `l1` and `l2`
    */
    layer Add(layer l1, layer l2);

    /**
      *  @brief Layer that computes the sum of a float number and a layer.
      *
      *  @param l1  Parent layer
      *  @param k  Number
      *  @return     Parent layer `l1` after computing its sum with `k`
    */
    layer Add(layer l1, float k);
    layer Add(float k, layer l1);

    // Deprecated
    layer Sum(layer l1, layer l2);
    layer Sum(layer l1, float k);
    layer Sum(float k, layer l1);

    /**
      *  @brief Returns a new layer which indexes the input tensor using the entries in indices
      *
      *  @param l  Parent layer
      *  @param indices  Vector of indices to be selected
      *  @param name  A name for the operation
      *  @return     A tensor with the selected elements
    */
    layer Select(layer l, vector<string> indices, string name="");

    /**
      *  @brief Returns a new layer which indexes the input tensor using the entries in indices. (alias for Select)
      *  Alias for Select
      *
      *  @param l  Parent layer
      *  @param indices  Vector of indices to be selected
      *  @param name  A name for the operation
      *  @return     A tensor with the selected elements
    */
    layer Slice(layer l, vector<string> indices, string name="");

    /**
      *  @brief Returns a layer with singleton dimensions expanded to a larger size.
      *  @param l  Parent layer
      *  @param size  Size to which expand the singleton dimensions
      *  @param name  A name for the operation
      *  @return     A tensor with the selected elements
    */
        layer Expand(layer l, int size, string name="");

    /**
      *  @brief Permutes the dimensions of the input according to a given pattern.
      *
      *  @param l  Parent layer
      *  @param dims  Permutation pattern, does not include the samples dimension.
      *  @param name  A name for the operation
      *  @return     The permuted tensor.
    */
    layer Permute(layer l, vector<int> dims, string name="");

    /**
      *  @brief Split a layer into a list of tensors layers
      *
      *  @param l  Parent layer
      *  @param indexes  Split indexes ({20, 60} => {0:20, 20:60, 60:end})
      *  @param axis  Which axis to split on (default=-1, last)
      *  @param merge_sublayers  Merge layers symbolically (for the plot)
      *  @param name  A name for the operation
      *  @return     vector of layers
    */
    vlayer Split(layer l, vector<int> indexes, int axis=-1, bool merge_sublayers=false, string name="");

    // Reduction Layers

    /**
      *  @brief Computes the mean of the elements over the given axis
      *
      *  @param l  Parent layer
      *  @param axis  Axis where to perform the reduction
      *  @param keepdims  Boolean flag to indicate if original dimensions must be preserved
      *  @return     A ReduceMean Layer.
    */
    layer ReduceMean(layer l, vector<int> axis, bool keepdims = false);

    /**
      *  @brief Computes the variance of the elements over the given axis
      *
      *  @param l  Parent layer
      *  @param axis  Axis where to perform the reduction
      *  @param keepdims  Boolean flag to indicate if original dimensions must be preserved
      *  @return     A ReduceVar Layer.
    */
    layer ReduceVar(layer l, vector<int> axis, bool keepdims = false);

    /**
      *  @brief Computes the sum of the elements over the given axis
      *
      *  @param l  Parent layer
      *  @param axis  Axis where to perform the reduction
      *  @param keepdims  Boolean flag to indicate if original dimensions must be preserved
      *  @return     A ReduceSum Layer.
    */
    layer ReduceSum(layer l, vector<int> axis, bool keepdims = false);

    /**
      *  @brief Computes the maximum of the elements over the given axis
      *
      *  @param l  Parent layer
      *  @param axis  Axis where to perform the reduction
      *  @param keepdims  Boolean flag to indicate if original dimensions must be preserved
      *  @return     A ReduceMax Layer.
    */
    layer ReduceMax(layer l, vector<int> axis, bool keepdims = false);

    /**
      *  @brief Computes the minimum of the elements over the given axis
      *
      *  @param l  Parent layer
      *  @param axis  Axis where to perform the reduction
      *  @param keepdims  Boolean flag to indicate if original dimensions must be preserved
      *  @return     A ReduceMin Layer.
    */
    layer ReduceMin(layer l, vector<int> axis, bool keepdims = false);

    /**
      *  @brief Computes the position of the maximum of the elements over the given axis
      *
      *  @param l  Parent layer
      *  @param axis  Axis where to perform the reduction
      *  @param keepdims  Boolean flag to indicate if original dimensions must be preserved
      *  @return     A ReduceArgMax Layer.
    */
    layer ReduceArgMax(layer l, vector<int> axis, bool keepdims = false);

    // Generator Layers

    /**
      *  @brief Generates a gaussian noise output (typically used for GANs) with the specified mean and standard deviation.
      *
      *  @param mean  Mean of the gaussian distribution
      *  @param stdev  Standard deviation of the gaussian distribution
      *  @param size  Shape of the output tensor of the layer
      *  @return     A layer that generates tensors with the specified gaussian distribution.
    */
    layer GaussGenerator(float mean, float stdev, vector<int> size);

    /**
      *  @brief Generates a uniform noise output (typically used for GANs) with the specified lower and upper bound values.
      *
      *  @param low  Lower bound of the uniform distribution
      *  @param high  Upper bound of the uniform distribution
      *  @param size  Shape of the output tensor of the layer
      *  @return     A layer that generates tensors with the specified uniform distribution.
    */
    layer UniformGenerator(float low, float high, vector<int> size);

    // Pooling Layers
    /**
      *  @brief Max pooling operation.
      *
      *  @param parent  Parent layer
      *  @param pool_size  Size of the max pooling windows
      *  @param strides  Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size
      *  @param padding  One of "none", "valid" or "same" (case-insensitive)
      *  @param name  A name for the operation
      *  @return     The result after applying the max pooling operation over the parent layer
    */
    layer MaxPool(layer parent, const vector<int> &pool_size = {2, 2}, const vector<int> &strides = {2, 2}, string padding = "none", string name = "");  // TODO: Deprecated? Generic but not generic... (2D only)

    /**
      *  @brief MaxPooling1D operation.
      *
      *  @param parent  Parent layer
      *  @param pool_size  Size of the max pooling windows
      *  @param strides  Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size
      *  @param padding  One of "none", "valid" or "same" (case-insensitive)
      *  @param name  A name for the operation
      *  @return     The result after applying the max pooling operation over the parent layer
    */
    layer MaxPool1D(layer parent, vector<int> pool_size = {2}, vector<int> strides = {2}, string padding = "none", string name = "");

    /**
      *  @brief MaxPooling2D operation.
      *
      *  @param parent  Parent layer
      *  @param pool_size  Size of the max pooling windows
      *  @param strides  Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size
      *  @param padding  One of "none", "valid" or "same" (case-insensitive)
      *  @param name  A name for the operation
      *  @return     The result after applying the max pooling operation over the parent layer
    */
    layer MaxPool2D(layer parent, vector<int> pool_size = {2, 2}, vector<int> strides = {2, 2}, string padding = "none", string name = "");

    /**
      *  @brief MaxPooling3D operation.
      *
      *  @param parent  Parent layer
      *  @param pool_size  Size of the max pooling windows
      *  @param strides  Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size
      *  @param padding  One of "none", "valid" or "same" (case-insensitive)
      *  @param name  A name for the operation
      *  @return     The result after applying the max pooling operation over the parent layer
    */
    layer MaxPool3D(layer parent, vector<int> pool_size = {2, 2, 2}, vector<int> strides = {2, 2, 2}, string padding = "none", string name = "");

    /**
     *  @brief Average pooling operation.
     *
     *  @param parent  Parent layer
     *  @param pool_size  Size of the average pooling windows
     *  @param strides  Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size
     *  @param padding  One of "none", "valid" or "same" (case-insensitive)
     *  @param name  A name for the operation
     *  @return     The result after apply the average pooling operation over the parent layer
   */
    layer AveragePool(layer parent, const vector<int> &pool_size = {2, 2}, const vector<int> &strides = {2, 2},string padding = "none", string name = "");
    layer AvgPool(layer parent, const vector<int> &pool_size = {2, 2}, const vector<int> &strides = {2, 2},string padding = "none", string name = "");  // Alias

    /**
      *  @brief AveragePooling1D operation.
      *
      *  @param parent  Parent layer
      *  @param pool_size  Size of the average pooling windows
      *  @param strides  Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size
      *  @param padding  One of "none", "valid" or "same" (case-insensitive)
      *  @param name  A name for the operation
      *  @return     The result after apply the average pooling operation over the parent layer
    */
    layer AveragePool1D(layer parent, vector<int> pool_size = {2}, vector<int> strides = {2}, string padding = "none", string name = "");
    layer AvgPool1D(layer parent, vector<int> pool_size = {2}, vector<int> strides = {2}, string padding = "none", string name = "");  // Alias

    /**
      *  @brief AveragePooling2D operation.
      *
      *  @param parent  Parent layer
      *  @param pool_size  Size of the average pooling windows
      *  @param strides  Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size
      *  @param padding  One of "none", "valid" or "same" (case-insensitive)
      *  @param name  A name for the operation
      *  @return     The result after apply the average pooling operation over the parent layer
    */
    layer AveragePool2D(layer parent, vector<int> pool_size = {2, 2}, vector<int> strides = {2, 2}, string padding = "none", string name = "");
    layer AvgPool2D(layer parent, vector<int> pool_size = {2, 2}, vector<int> strides = {2, 2}, string padding = "none", string name = "");  // Alias

    /**
  *  @brief AveragePooling3D operation.
  *
  *  @param parent  Parent layer
  *  @param pool_size  Size of the average pooling windows
  *  @param strides  Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size
  *  @param padding  One of "none", "valid" or "same" (case-insensitive)
  *  @param name  A name for the operation
  *  @return     The result after apply the average pooling operation over the parent layer
*/
    layer AveragePool3D(layer parent, vector<int> pool_size = {2, 2, 2}, vector<int> strides = {2, 2, 2}, string padding = "none", string name = "");
    layer AvgPool3D(layer parent, vector<int> pool_size = {2, 2, 2}, vector<int> strides = {2, 2, 2}, string padding = "none", string name = "");  // Alias

    /**
      *  @brief GlobalMax pooling operation.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     The result after applying the global max pooling operation over the parent layer
    */
    layer GlobalMaxPool(layer parent, string name = "");  // TODO: Deprecated? Generic but not generic... (2D only)

    /**
    *  @brief GlobalMaxPooling1D operation.
    *
    *  @param parent  Parent layer
    *  @param name  A name for the operation
    *  @return     The result after applying the global max pooling operation over the parent layer
*/
    layer GlobalMaxPool1D(layer parent, string name = "");


    /**
        *  @brief GlobalMaxPooling2D operation.
        *
        *  @param parent  Parent layer
        *  @param name  A name for the operation
        *  @return     The result after applying the global max pooling operation over the parent layer
    */
    layer GlobalMaxPool2D(layer parent, string name = "");

    /**
        *  @brief GlobalMaxPooling3D operation.
        *
        *  @param parent  Parent layer
        *  @param name  A name for the operation
        *  @return     The result after applying the global max pooling operation over the parent layer
    */
    layer GlobalMaxPool3D(layer parent, string name = "");

    /**
      *  @brief GlobalAveragePooling operation.
      *
      *  @param parent  Parent layer
      *  @param name  A name for the operation
      *  @return     The result after applying the global average pooling operation over the parent layer
    */
    layer GlobalAveragePool(layer parent, string name = "");
    layer GlobalAvgPool(layer parent, string name = "");  // Alias

    /**
        *  @brief GlobalAveragePooling1D operation.
        *
        *  @param parent  Parent layer
        *  @param name  A name for the operation
        *  @return     The result after applying the global average pooling operation over the parent layer
    */
    layer GlobalAveragePool1D(layer parent, string name = "");
    layer GlobalAvgPool1D(layer parent, string name = "");  // Alias


    /**
        *  @brief GlobalAveragePooling2D operation.
        *
        *  @param parent  Parent layer
        *  @param name  A name for the operation
        *  @return     The result after applying the global average pooling operation over the parent layer
    */
    layer GlobalAveragePool2D(layer parent, string name = "");
    layer GlobalAvgPool2D(layer parent, string name = "");  // Alias

    /**
        *  @brief GlobalAveragePooling3D operation.
        *
        *  @param parent  Parent layer
        *  @param name  A name for the operation
        *  @return     The result after applying the global average pooling operation over the parent layer
    */
    layer GlobalAveragePool3D(layer parent, string name = "");
    layer GlobalAvgPool3D(layer parent, string name = "");

    // Recurrent Layers

    /**
      *  @brief Fully-connected RNN where the output is to be fed back to input.
      *
      *  @param parent  Parent layer
      *  @param units  Dimensionality of the output space
      *  @param activation Name of the activation function
      *  @param use_bias  Whether the layer uses a bias vector
      *  @param bidirectional  Wether the RNN is bidirectional or not.
      *  @param name  A name for the operation
      *  @return     The RNN layer
    */
    layer RNN(layer parent, int units, string activation="tanh", bool use_bias = true, bool bidirectional = false, string name = "");

    /**
      *  @brief Long Short-Term Memory layer - Hochreiter 1997.
      *
      *  @param parent  Parent layer
      *  @param units  Dimensionality of the output space
      *  @param mask_zeros
      *  @param bidirectional  Wether the RNN is bidirectional or not
      *  @param name  A name for the operation
      *  @return     The LSTM layer
    */
    layer LSTM(layer parent, int units, bool mask_zeros=false, bool bidirectional = false, string name = "");

    layer LSTM(vector<layer> parent, int units, bool mask_zeros=false, bool bidirectional = false, string name = "");

    layer States(const vector<int> &shape, string name = "");

    /**
      *  @brief Gated Recurrent Unit (GRU).
      *
      *  @param parent  Parent layer
      *  @param units  Dimensionality of the output space
      *  @param mask_zeros
      *  @param bidirectional  Wether the RNN is bidirectional or not
      *  @param name  A name for the operation
      *  @return     The GRU layer
    */
    layer GRU(layer parent, int units, bool mask_zeros=false, bool bidirectional = false, string name = "");

    layer GRU(vector<layer> parent, int units, bool mask_zeros=false, bool bidirectional = false, string name = "");

    layer GetStates(layer parent);

    void setDecoder(layer l);

    // Layers Methods
    vlayer getOut(model net);

    // Manage tensors inside layers
    Tensor* getOutput(layer l1);
    Tensor* getInput(layer l1);
    Tensor* getDelta(layer l1);
    Tensor* getParam(layer l1, int p);
    Tensor* getGradient(layer l1,int p);
    Tensor* getState(layer l1,int p);
    vector<Tensor*> getParams(layer l1);
    vector<Tensor*> getGradients(layer l1);
    vector<Tensor*> getStates(layer l1);
    void copyOutput(Layer *l1,Layer *l2);
    void copyDelta(Layer *l1,Layer *l2);
    void copyParam(Layer *l1,Layer *l2, int p=-1);
    void copyGradient(Layer *l1,Layer *l2, int p);
    void distributeParams(Layer *l);

    ///////////////////////////////////////
    //  INITIALIZERS
    ///////////////////////////////////////
    /**
      *  @brief Glorot normal initializer, also called Xavier normal initializer.
      *
      *  @details
      *   It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor.
      *
      *  @param l  Parent layer to initialize
      *  @param seed   Used to seed the random generator
      *  @return     The layer l initialized with the Glorot normal
    */
    layer GlorotNormal(layer l,int seed=1234);
    /**
      *  @brief Glorot uniform initializer, also called Xavier uniform initializer.
      *
      *  @details
      *   It draws samples from a uniform distribution within [-limit, limit] where limit is sqrt(6 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor.
      *
      *  @param l  Parent layer to initialize
      *  @param seed   Used to seed the random generator
      *  @return     The layer l initialized with the Glorot uniform
    */
    layer GlorotUniform(layer l,int seed=1234);

    /**
      *  @brief He uniform initializer
      *
      *  @details
      *   It draws samples from a uniform distribution within [-limit, limit] where limit is sqrt(6 / (fan_in )) where fan_in is the number of input units in the weight tensor
      *
      *  @param l  Parent layer to initialize
      *  @param seed   Used to seed the random generator
      *  @return     The layer l initialized with the Glorot uniform
    */
    layer HeUniform(layer l,int seed=1234);

    /**
      *  @brief He normal initializer
      *
      *  @details
      *   It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (fan_in)) where fan_in is the number of input units in the weight tensor
      *
      *  @param l  Parent layer to initialize
      *  @param seed   Used to seed the random generator
      *  @return     The layer l initialized with the Glorot normal
    */
    layer HeNormal(layer l,int seed=1234);


    /**
      *  @brief Random normal initializer.
      *
      *  @param l  Parent layer to initialize
      *  @param m  Mean of the normal distribution to draw samples
      *  @param s  Standard deviation of the normal distribution to draw samples
      *  @param seed   Used to seed the random generator
      *  @return     The layer l initialized with a random normal distribution
    */
    layer RandomNormal(layer l, float m=0.0,float s=0.1, float seed=1234);

    /**
      *  @brief Random uniform initializer.
      *
      *  @param l  Parent layer to initialize
      *  @param min lower bound of the uniform distribution
      *  @param max upper bount of the uniform distribution
      *  @param seed   Used to seed the random generator
      *  @return     The layer l initialized with a random normal distribution
    */
    layer RandomUniform(layer l, float min=0.0,float max=0.1, float seed=1234);

    /**
      *  @brief Initializer that generates tensors initialized to a constant value.
      *
      *  @param l  Parent layer to initialize
      *  @param v   Value of the generator
      *  @return     The layer l initialized with a constant value
    */
    layer Constant(layer l, float v=0.1);


    ///////////////////////////////////////
    //  REGULARIZERS
    ///////////////////////////////////////

    /**
      *  @brief Regularizer for L2 regularization.
      *
      *  @param l  Parent layer to regularize
      *  @param l2   L2 regularization factor
      *  @return     The layer `l` regularized
    */
    layer L2(layer l,float l2);

    /**
      *  @brief Regularizer for L1 regularization.
      *
      *  @param l  Parent layer to regularize
      *  @param l1   L1 regularization factor
      *  @return     The layer `l` regularized
    */
    layer L1(layer l,float l1);

    /**
      *  @brief Regularizer for L1 and L2 regularization.
      *
      *  @param l  Parent layer to regularize
      *  @param l1   L1 regularization factor
      *  @param l2   L2 regularization factor
      *  @return     The layer `l` regularized
    */
    layer L1L2(layer l,float l1,float l2);

    ///////////////////////////////////////
    //  FUSED LAYERS
    ///////////////////////////////////////
    /**
      *  @brief Convolution + Activation layer.
      *
      *  @param parent  Parent layer
      *  @param filters  Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)
      *  @param kernel_size  Vector of 2 integers, specifying the height and width of the 2D convolution window
      *  @param strides  Vector of 2 integers, specifying the strides of the convolution along the height and width
      *  @param padding  One of "none", "valid" or "same"
      *  @param use_bias  Boolean, whether the layer uses a bias vector
      *  @param groups  Number of blocked connections from input channels to output channels
      *  @param dilation_rate  Vector of 2 integers, specifying the dilation rate to use for dilated convolution
      *  @param name  A name for the operation
      *  @return     Convolution layer
    */
    layer Conv2dActivation(layer parent, string act, int filters, const vector<int> &kernel_size,
                           const vector<int> &strides = {1, 1}, string padding = "same", bool use_bias = true,
                           int groups = 1, const vector<int> &dilation_rate = {1, 1}, string name = "");

    ///////////////////////////////////////
    // MODELS
    ///////////////////////////////////////
    void download_model(string name,string link);

    /**
      *  @brief Returns a VGG16 model pretrained with imagenet.
      *
      *  @param top  If true, returns the model without the densely connected part and
      *              the last layer of the returned model is named "top".
      *  @param input_shape  Optional. To change the input shape of the model.
      *                      The shape vector must not have the batch dimension.
      *  @return  A VGG16 Net* with the desired topology
    */
    Net* download_vgg16(bool top=true, vector<int> input_shape={});

    /**
      *  @brief Returns a VGG16 model with BatchNormalization pretrained with imagenet.
      *
      *  @param top  If true, returns the model without the densely connected part and
      *              the last layer of the returned model is named "top".
      *  @param input_shape  Optional. To change the input shape of the model.
      *                      The shape vector must not have the batch dimension.
      *  @return  A VGG16-BN Net* with the desired topology
    */
    Net* download_vgg16_bn(bool top=true, vector<int> input_shape={});

    /**
      *  @brief Returns a VGG19 model pretrained with imagenet.
      *
      *  @param top  If true, returns the model without the densely connected part and
      *              the last layer of the returned model is named "top".
      *  @param input_shape  Optional. To change the input shape of the model.
      *                      The shape vector must not have the batch dimension.
      *  @return  A VGG19 Net* with the desired topology
    */
    Net* download_vgg19(bool top=true, vector<int> input_shape={});

    /**
      *  @brief Returns a VGG19 model with BatchNormalization pretrained with imagenet.
      *
      *  @param top  If true, returns the model without the densely connected part and
      *              the last layer of the returned model is named "top".
      *  @param input_shape  Optional. To change the input shape of the model.
      *                      The shape vector must not have the batch dimension.
      *  @return  A VGG19-BN Net* with the desired topology
    */
    Net* download_vgg19_bn(bool top=true, vector<int> input_shape={});

    /**
      *  @brief Returns a ResNet18 model pretrained with imagenet.
      *
      *  @param top  If true, returns the model without the densely connected part and
      *              the last layer of the returned model is named "top".
      *  @param input_shape  Optional. To change the input shape of the model.
      *                      The shape vector must not have the batch dimension.
      *  @return  A ResNet18 Net* with the desired topology
    */
    Net* download_resnet18(bool top=true, vector<int> input_shape={});

    /**
      *  @brief Returns a ResNet34 model pretrained with imagenet.
      *
      *  @param top  If true, returns the model without the densely connected part and
      *              the last layer of the returned model is named "top".
      *  @param input_shape  Optional. To change the input shape of the model.
      *                      The shape vector must not have the batch dimension.
      *  @return  A ResNet34 Net* with the desired topology
    */
    Net* download_resnet34(bool top=true, vector<int> input_shape={}); 

    /**
      *  @brief Returns a ResNet50 model pretrained with imagenet.
      *
      *  @param top  If true, returns the model without the densely connected part and
      *              the last layer of the returned model is named "top".
      *  @param input_shape  Optional. To change the input shape of the model.
      *                      The shape vector must not have the batch dimension.
      *  @return  A ResNet50 Net* with the desired topology
    */
    Net* download_resnet50(bool top=true, vector<int> input_shape={}); 

    /**
      *  @brief Returns a ResNet101 model pretrained with imagenet.
      *
      *  @param top  If true, returns the model without the densely connected part and
      *              the last layer of the returned model is named "top".
      *  @param input_shape  Optional. To change the input shape of the model.
      *                      The shape vector must not have the batch dimension.
      *  @return  A ResNet101 Net* with the desired topology
    */
    Net* download_resnet101(bool top=true, vector<int> input_shape={});

    /**
      *  @brief Returns a ResNet152 model pretrained with imagenet.
      *
      *  @param top  If true, returns the model without the densely connected part and
      *              the last layer of the returned model is named "top".
      *  @param input_shape  Optional. To change the input shape of the model.
      *                      The shape vector must not have the batch dimension.
      *  @return  A ResNet152 Net* with the desired topology
    */
    Net* download_resnet152(bool top=true, vector<int> input_shape={});

    /**
      *  @brief Returns a DenseNet121 model pretrained with imagenet.
      *
      *  @param top  If true, returns the feature extraction part and
      *              the last layer of the returned model is named "top".
      *  @param input_shape  Optional. To change the input shape of the model.
      *                      The shape vector must not have the batch dimension.
      *  @return  A DenseNet121 Net* with the desired topology
    */
    Net* download_densenet121(bool top=true, vector<int> input_shape={});

    ///////////////////////////////////////
    //  DATASETS
    ///////////////////////////////////////
    bool exist(string name);
    /**
      *  @brief Downloads MNIST Dataset.
      *
      *  @see   http://yann.lecun.com/exdb/mnist/
      *
      *  @return     (void) The binary files of MNIST
    */
    void download_mnist();
    /**
      *  @brief Downloads CIFAR-10 Dataset.
      *
      *  @see   https://www.cs.toronto.edu/~kriz/cifar.html
      *
      *  @return     (void) The binary files of CIFAR-10
    */
    void download_cifar10();
    /**
      *  @brief Downloads DRIVE Dataset.
      *
      *  @see   https://drive.grand-challenge.org/
      *
      *  @return     (void) The numpy files of DRIVE
    */
    void download_drive();

    /**
      *  @brief Downloads IMDB Dataset. 2000 most frequent words
      *
      *  @see   https://ai.stanford.edu/~amaas/data/sentiment/
      *
      *  @return     (void) The binary files of IMDB
    */
    void download_imdb_2000();


    /**
      *  @brief Downloads EuTrans Dataset.
      *
      *  @see
      *
      *  @return     (void) The binary files of EuTrans
    */
    void download_eutrans();

    /**
      *  @brief Downloads Flickr Dataset (small partition)
      *
      *  @see
      *
      *  @return     (void) The binary files of Flickr
    */
    void download_flickr();

    // Auxiliary function
    layer _expand3d_to_4d(layer parent, string name);
}
#endif
