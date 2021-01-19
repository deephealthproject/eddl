/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/core/layer_core.h"

using namespace std;

int LActivation::total_layers = 0;

LActivation::LActivation(Layer *parent, string act, vector<float> params, string name, int dev, int mem) : LinLayer(name, dev, mem){
    // Set default name
    if(name.empty()) this->name = act + to_string(++total_layers);

    this->act = act;
    this->params = params;

    input = parent->output;
#ifdef DEBUG_FPGA
    printf("creating output for RELU\n");
#endif
    output = new Tensor(input->shape, dev);
    delta_bp = 0;

#ifdef cCUDNN

    data_type = CUDNN_DATA_FLOAT;
    tensor_format = CUDNN_TENSOR_NCHW;  // CUDNN_TENSOR_NHWC
    //BOTH softmax and activations
    cudnn_handle = hdnn;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnSetTensor4dDescriptor(xDesc, tensor_format, data_type,
                 input->shape[0], input->shape[1], input->shape[2], input->shape[3]);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnSetTensor4dDescriptor(yDesc, tensor_format, data_type,
                 output->shape[0], output->shape[1], output->shape[2], output->shape[3]);


    if(this->act == "softmax"){
        algorithm = CUDNN_SOFTMAX_ACCURATE;
        softmax_mode = CUDNN_SOFTMAX_MODE_INSTANCE;

    }
    else{
        cudnnCreateActivationDescriptor(&activationDesc);
        if(this->act == "sigmoid"){
            mode = CUDNN_ACTIVATION_SIGMOID;
        }
        else if(this->act == "relu"){
            mode = CUDNN_ACTIVATION_RELU;
            coef = 9999999.0f; //this->params[0]; //upper boud
            reluNanOpt = CUDNN_PROPAGATE_NAN;
        }
        else if(this->act == "thresholded_relu"){
            mode = CUDNN_ACTIVATION_CLIPPED_RELU;
            coef = this->params[0]; //threshold
            reluNanOpt = CUDNN_PROPAGATE_NAN;
        }
        else if(this->act == "tanh"){
            mode = CUDNN_ACTIVATION_TANH;
        }
        else if(this->act == "elu"){
            mode = CUDNN_ACTIVATION_ELU;
            reluNanOpt = CUDNN_PROPAGATE_NAN;
        }
        else if(this->act == "linear"){
            mode = CUDNN_ACTIVATION_IDENTITY;
        }
        else{
            std:cerr<<"Warning. "<<this->act<<" activation is not supported in CUDNN. A RELU will be executed." <<std::endl;
            mode = CUDNN_ACTIVATION_RELU;
            coef = 9999999.0f; //this->params[0]; //upper boud
            reluNanOpt = CUDNN_PROPAGATE_NAN;
        }

        cudnnSetActivationDescriptor( activationDesc, mode, reluNanOpt, coef);
    }


#else

    // Softmax checks
    if(this->act=="softmax"){
        // Set default axis if none was specified
        if(this->params.empty()){
            this->params.push_back(-1);
            std::cerr << "No axis for 'softmax' was specified. Using last one (-1) as default " << "(LActivation::Softmax)" << endl;
        }

        // Check number of axis
        if(this->params.size()>1){
            msg("Only one axis is supported ("  + std::to_string(this->params.size()) + " were specified)", "LActivation::Softmax");
        }

        // Replace -1 axis with last one
        int lastAxis = (int)input->shape.size()-1;
        if((int)this->params[0]==-1){
            this->params[0] = lastAxis;
        }

        // Check bounds
        if((int)this->params[0] <0 || (int)this->params[0]>lastAxis){
            msg("The axis has to be a number from 0 to (number_of_dimensions - 1)", "LActivation::Softmax");
        }
    }
#endif

    parent->addchild(this);
    addparent(parent);
}


void LActivation::forward(){

#ifdef cCUDNN
    float alpha = 1.0f;
    float beta = 0.0f;
    if(act == "softmax"){
        cudnnSoftmaxForward(cudnn_handle, algorithm, softmax_mode, &alpha, xDesc, input, &beta, yDesc, output);
    }
    else{
        cudnnActivationForward(cudnn_handle, activationDesc, &alpha, xDesc, input, &beta, yDesc, output);
    }
#else
    if (act == "relu"){
        tensorNN::ReLu(this->input, this->output);
    }else if (act == "thresholded_relu"){
        float alpha = this->params[0];
        tensorNN::ThresholdedReLu(this->input, this->output, alpha);

    }else if (act == "elu"){
        float alpha = this->params[0];
        tensorNN::ELu(this->input, this->output, alpha);

    }else if (act == "selu"){
        // https://mlfromscratch.com/activation-functions-explained/#selu
        float alpha = this->params[0];
        float scale = this->params[1];

        tensorNN::ELu(this->input, this->output, alpha);
        this->output->mult_(scale);

    }else if (act == "exp"){
        tensorNN::Exp(this->input, this->output);

    }else if (act == "softplus"){
        tensorNN::Softplus(this->input, this->output);

    }else if (act == "softsign"){
        tensorNN::Softsign(this->input, this->output);

    }else if (act == "softmax_deprecated"){  // TODO: Deprecated
        tensorNN::Softmax(this->input, this->output);

    }else if (act == "softmax"){
        int axis = (int)this->params[0];
        tensorNN::FullSoftmax(this->input, this->output, axis);

    }else if (act == "sigmoid"){
        tensorNN::Sigmoid(this->input, this->output);

    }else if (act == "hard_sigmoid"){
        tensorNN::HardSigmoid(this->input, this->output);

    }else if (act == "leaky_relu"){
        float alpha = this->params[0];
        tensorNN::LeakyReLu(this->input, this->output, alpha);

    }else if (act == "tanh"){
        tensorNN::Tanh(this->input, this->output);

    }else if (act == "linear"){
        float alpha = this->params[0];
        tensorNN::Linear(this->input, this->output, alpha);
    }
#endif
}


void LActivation::backward(){
    if (delta_bp){
        Tensor::inc(delta, parent[0]->delta);
    }else {
        if (act == "relu"){
            tensorNN::D_ReLu(delta, input, parent[0]->delta);

        }else if (act == "thresholded_relu"){
            float alpha = this->params[0];
            tensorNN::D_ThresholdedReLu(delta, input, parent[0]->delta, alpha);

        }else if (act == "elu"){
            float alpha = this->params[0];
            tensorNN::D_ELu(delta, input, parent[0]->delta, alpha);

        }else if (act == "selu"){
            // https://mlfromscratch.com/activation-functions-explained/#selu
            float alpha = this->params[0];
            float scale = this->params[1];

            tensorNN::D_ELu(delta, input, parent[0]->delta, alpha);
            this->output->mult_(scale);

        }else if (act == "exp"){
            tensorNN::D_Exp(delta, output, parent[0]->delta);

        }else if (act == "softplus"){
            tensorNN::D_softplus(delta, output, parent[0]->delta);

        }else if (act == "softsign"){
            tensorNN::D_softsign(delta, output, parent[0]->delta);

        }else if (act == "softmax_deprecated"){  // TODO: Deprecaated
            tensorNN::D_Softmax(delta, output, parent[0]->delta);

        }else if (act == "softmax"){
            int axis = (int)this->params[0];
            tensorNN::D_FullSoftmax(delta, output, parent[0]->delta, axis);

        }else if (act == "sigmoid"){
            tensorNN::D_Sigmoid(delta, output, parent[0]->delta);

        }else if (act == "hard_sigmoid"){
            tensorNN::D_HardSigmoid(delta, input, parent[0]->delta);

        }else if (act == "leaky_relu"){
            float alpha = this->params[0];
            tensorNN::D_LeakyReLu(delta, input, parent[0]->delta, alpha);

        }else if (act == "tanh"){
            tensorNN::D_Tanh(delta, output, parent[0]->delta);

        }else if (act == "linear"){
            float alpha = this->params[0];
            tensorNN::D_Linear(delta, input, parent[0]->delta, alpha);
        }
    }
}


void LActivation::save(std::ofstream &ofs, string format){
    // Save act
    // Save param for "lrelu"
}

void LActivation::load(std::ifstream &ifs, string format){
    // Load act
    // Load param for "lrelu"
}

Layer *LActivation::share(int c, int bs, vector<Layer *> p){
    LActivation *n = new LActivation(p[0], this->act, this->params, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;
    n->delta_bp = delta_bp;

    return n;
}

Layer *LActivation::clone(int c, int bs, vector<Layer *> p, int todev){

    LActivation *n = new LActivation(p[0], this->act, this->params,  "clone_" + name, todev, this->mem_level);
    n->orig = this;
    n->delta_bp = delta_bp;

    return n;
}


string LActivation::plot(int c){
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=LightBlue,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=LightSalmon,shape=box]";

    return s;
}
