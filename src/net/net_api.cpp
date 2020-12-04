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
#include <fstream>
#include <string>
#include <chrono>
#include <stdexcept>
#include "eddl/layers/core/layer_core.h"
#include "eddl/net/net.h"
#include "eddl/random.h"
#include "eddl/system_info.h"
#include "eddl/utils.h"


#ifdef cFPGA
extern void _show_profile_fpga();
#endif

#define VERBOSE 0

using namespace std;
using namespace std::chrono;

/////////////////////////////////////////
//// THREADS
struct tdata {
  Net *net;
};

/////////////////////////////////////////
void *train_batch_t(void *t) {
  auto *targs = (tdata *) t;

  Net *net = targs->net;
  net->do_reset();
  net->do_reset_grads();
  net->do_forward();
  net->do_compute_loss();

  net->do_delta();
  net->do_backward();
  net->do_applygrads();

  return nullptr;
}

void *eval_batch_t(void *t) {
  auto *targs = (tdata *) t;

  Net *net = targs->net;
  net->do_reset();
  net->do_reset_grads();
  net->do_forward();
  net->do_compute_loss();

  return nullptr;
}

/////////////////////////////////////////
void *forward_t(void *t) {
  auto *targs = (tdata *) t;

  Net *net = targs->net;

  net->do_forward();

  return nullptr;
}

/////////////////////////////////////////
void *reset_t(void *t) {
  auto *targs = (tdata *) t;

  Net *net = targs->net;

  net->do_reset();

  return nullptr;
}
void *reset_grads_t(void *t) {
  auto *targs = (tdata *) t;

  Net *net = targs->net;

  net->do_reset_grads();

  return nullptr;
}

/////////////////////////////////////////

void *delta_t(void *t) {
  auto *targs = (tdata *) t;

  Net *net = targs->net;

  net->do_delta();

  return nullptr;
}
/////////////////////////////////////////
void *backward_t(void *t) {
  auto *targs = (tdata *) t;

  Net *net = targs->net;


  net->do_delta();
  net->do_backward();

  return nullptr;
}

void *compute_loss_t(void *t)
{
  auto *targs = (tdata *) t;

  Net *net = targs->net;

  net->do_compute_loss();

  return nullptr;
}

/////////////////////////////////////////
void *update_t(void *t) {
  auto *targs = (tdata *) t;

  Net *net = targs->net;
  net->do_applygrads();

  return nullptr;
}
/////////////////////////////////////////

/////////////////////////////////////////
// "a ring to rule them all"
void Net::run_snets(void *(*F)(void *t))
{
  void *status;
  int rc;
  struct tdata td[100];

  int comp = snets.size();

  #pragma omp parallel for
  for (int i = 0; i < comp; i++) {
    // Thread params
    td[i].net = snets[i];
    // Call function
    F(&td[i]);
  }
}


//////////////////////////////////////////////////////////////
//////// SIMPLE ATOMICS FUNCS
void Net::setmode(int m) {
  trmode=m;
  for (int i = 0; i < snets.size(); i++)
  for (int j = 0; j < snets[i]->layers.size(); j++)
  snets[i]->layers[j]->setmode(m);
}

void Net::clamp(float min,float max)
{
  for (int i = 0; i < snets.size(); i++)
  for (int j = 0; j < snets[i]->layers.size(); j++)
  snets[i]->layers[j]->clamp(min,max);
}


void Net::setlr(vector <float> p)
{
  for(int i=0;i<snets.size();i++)
  snets[i]->optimizer->change(p);
}

vector<vtensor> Net::get_parameters(bool deepcopy, bool tocpu){
    vector<vtensor> net_params;

    // Collect layer params
    for(auto &l : this->layers){
        if(!deepcopy){
            net_params.push_back(l->params);
        }else{
            // Clone parameters
            vtensor lp;
            for(auto &param : l->params){
                Tensor* new_param = param->clone();
                if(tocpu) { new_param->toCPU(); }  // Send to CPU
                lp.push_back(new_param);  // Add to layer vector of params
            }

            // Add new params
            net_params.push_back(lp);
        }
    }

    return net_params;
}

void Net::set_parameters(const vector<vtensor>& params){
    // Check the number of layers
    if(params.size() != this->layers.size()){
        msg("AssertionError: The number of layers in params does not match the number of layers in this network ("
        + std::to_string(params.size())  + "!=" + std::to_string(this->layers.size()) +")",
        "Net::set_parameters");
    }

    // Check the number of params per layer
    for(int i=0; i<this->layers.size(); i++){
        Layer* l = this->layers[i];  // Alias

        // Check number of params
        if(params[i].size() != l->params.size()){
            msg("AssertionError: The number of parameters in layer '" + std::to_string(i) +
            "'(" + l->name +") does not match the number of parameters in the given vector<Tensor*>. (" +
            std::to_string(l->params.size()) + "!=" + std::to_string(params[i].size())  +")",
                "Net::set_parameters");
        }
    }

    // Set layer params
    for(int i=0; i<this->layers.size(); i++){

        // Copy current params
        for(int j=0; j<this->layers[i]->params.size(); j++){
            Tensor* new_param = params[i][j];
            new_param->toDevice(this->dev);  // Send to the same device as the net
            Tensor::copy(new_param, this->layers[i]->params[j]);
        }

    }
}

//////////////////////////////////
// API functions

// FORWARD
void Net::forward(vector<Tensor*> in)
{

  if (isrecurrent) {
    forward_recurrent(in);
  }
  else {

    reset();
    if (in.size()) {
      if (in.size()!=lin.size())
      msg("size missmatch in list of tensors","Net.forward(vtensor)");

      if (batch_size!=in[0]->shape[0]) {
        resize(in[0]->shape[0]);
      }

      for (int i = 0; i < in.size(); i++) {
        Tensor::copy(in[i],lin[i]->output);
      }

      // Distribute to snets inputs
      for (int i = 0; i < in.size(); i++)
        distributeTensor(lin[i]);


    }

    run_snets(forward_t);
  }

}

void Net::forward_recurrent(vector<Tensor*> tin)
{
  int i,j,k,l;

  // prepare data for unroll net
  vtensor xt;
  vtensor xtd;
  vtensor yt;

  vtensor toutr;
  vtensor tinr;
  vtensor tout;

  int inl;
  int outl;


  prepare_recurrent(tin,tout,inl,outl,xt,xtd,yt,tinr,toutr);

  build_rnet(inl,outl);

  if ((isencoder)&&(isdecoder))
    rnet->forward(tinr);
  else if (isencoder)
    rnet->forward(tinr);
  else if (isdecoder)
    rnet->forward(tin);

  for(i=0;i<xt.size();i++)
    delete xt[i];
  xt.clear();

  for(i=0;i<xtd.size();i++)
    delete xtd[i];
  xtd.clear();

}


void Net::forward(vector<Layer *> in)
{
  netinput=in;

  reset();
  if (in.size()) {
    if (in.size()!=lin.size())
    msg("size missmatch in list of tensors","Net.forward(vtensor)");

    if (batch_size!=in[0]->output->shape[0]) {
      resize(in[0]->output->shape[0]);
    }
  }

  vector<Tensor *> vt;
  for (int i = 0; i < in.size(); i++) {
     collectTensor(in[i],"output");
     vt.push_back(in[i]->output);
   }


  forward(vt);

}

void Net::forward()
{
  reset();

  run_snets(forward_t);
}


//// BACKWARD
void Net::backward(vector<Tensor *> target)
{

  if (isrecurrent) {
    if (rnet==nullptr) {
      msg("Error backward without previous forward","backward_recurrent");
    }
    backward_recurrent(target);
  }
  else  {

    if (target.size()) {
      if (target.size()!=lout.size())
      msg("size missmatch in list of targets","Net.backward(vtensor)");

      if (batch_size!=target[0]->shape[0])
      msg("bakcward step with different batch_size than forward","Net.backward(vtensor)");

      int comp=snets.size();
      if (batch_size<comp) {
        msg("batch_size lower than computing service parallelism","backward");
      }

      int thread_batch_size=batch_size / comp;

      // Split data for each network
      for (int i = 0; i < comp; i++) {
        int start = i * thread_batch_size;
        int end = start + Ys[i][0]->shape[0];
        vector<int> sind(batch_size);
        for(int k=0;k<batch_size;k++) sind[k]=k;
        // Copy targets
        for (int j = 0; j < target.size(); j++) {
          Tensor::select(target[j], Ys[i][j], sind, start, end);
          snets[i]->lout[j]->check_target();
          Tensor::copy(Ys[i][j], snets[i]->lout[j]->target);
        }
      }
    }
    tr_batches++;

    compute_loss();

    run_snets(backward_t);

  }
}

void Net::backward_recurrent(vector<Tensor *> target)
{
  int i,j,k,l;

  // prepare data for unroll net
  vtensor xt;
  vtensor xtd;
  vtensor yt;

  vtensor toutr;
  vtensor tinr;
  vtensor tin;

  int inl;
  int outl;

  prepare_recurrent(tin,target,inl,outl,xt,xtd,yt,tinr,toutr);

  if ((isencoder)&&(isdecoder))
    rnet->backward(toutr);
  else if (isencoder)
    rnet->backward(target);
  else if (isdecoder)
    rnet->backward(toutr);

  for(i=0;i<xt.size();i++)
    delete xt[i];
  xt.clear();

  for(i=0;i<xtd.size();i++)
    delete xtd[i];
  xtd.clear();

}


void Net::backward(){

  vector<Net*> visited;
  tr_batches++;

  run_snets(backward_t);


  for(int i=0;i<netinput.size();i++) {
    if (netinput[i]->detached==false) {
      lin[i]->mem_delta();
      collectTensor(lin[i],"delta");
      netinput[i]->mem_delta();
      Tensor::copy(lin[i]->delta,netinput[i]->delta);
      distributeTensor(netinput[i],"delta");
    }
  }

  for(int i=0;i<netinput.size();i++) {
    if (netinput[i]->detached==false){
      bool enc=false;
      for(int j=0;j<visited.size();j++)
      if (visited[j]==netinput[i]->net) enc=true;

      if (!enc) {
        visited.push_back(netinput[i]->net);
        netinput[i]->net->backward();
      }
    }
  }

  netinput.clear();

}




//// Loss
void Net::reset_loss()
{
  if (isrecurrent) {
    if (rnet!=nullptr) rnet->reset_loss();
  }
  else {
    // Reset errors
    int p=0;
    for (int j = 0; j < lout.size(); j++,p+=2){
      total_loss[j] = 0.0;
      total_metric[j] = 0.0;
      fiterr[p] = fiterr[p + 1] = 0.0;
    }
    inferenced_samples=0;
  }
}


//// COMPUTE Loss
void Net::compute_loss()
{

  if (isrecurrent) {
    if (rnet==nullptr) {
      msg("Error compute loss unroll net","compute_loss");
    }
    rnet->compute_loss();
  }
  else {
    run_snets(compute_loss_t);

    int comp=snets.size();
    if (batch_size<comp) {
      msg("batch_size lower than computing service parallelism","compute_loss");
    }


    if (snets[0]->dev != DEV_CPU)
    for (int i = 0; i < comp; i++) {
      for (int j = 0; j < 2 * lout.size(); j++) {
        fiterr[j] += snets[i]->fiterr[j];
      }
    }

    inferenced_samples+=batch_size;
  }
}

float Net::get_metric( const string  layer_name, const string  metric_name )
{
    float value=-1.;
    string lname="";

    if (isrecurrent) {
        value = -1.;
    } else {
        int p=0;
        int length = decsize;
        for (int k = 0; k < lout.size(); k+=decsize) {

            for( int l=0; l < length; l++, p+=2 ) {
                total_loss[k] += fiterr[p];  // loss
                total_metric[k] += fiterr[p + 1];  // metric
                fiterr[p] = fiterr[p + 1] = 0.0;
            }

            if ( layer_name.size() > 0 ) {
                lname = lout[k]->name;
                if (lout[k]->isshared) lname=lout[k]->orig->name;
            }

            // if no layer specified and more than one layer then
            // the required metric of the last output layer will be returned

            if ( layer_name.size() == 0 || layer_name == lname ) {

                if ( losses[k]->name == metric_name ) {
                    value = total_loss[k] / (length*inferenced_samples);
                } else if ( metrics[k]->name == metric_name ) {
                    value = total_metric[k] / (length*inferenced_samples);
                }
            }
        }
    }
    return value;
}

void Net::print_loss(int b)
{
  int p = 0;

  if (isrecurrent) {
    if (rnet!=nullptr) rnet->print_loss(b);
  }
  else {
    fprintf(stdout,"Batch %d ",b);
    int length=decsize;
    for (int k = 0; k < lout.size(); k+=decsize) {

      for(int l=0;l<length;l++,p+=2) {
        total_loss[k] += fiterr[p];  // loss
        total_metric[k] += fiterr[p + 1];  // metric
        fiterr[p] = fiterr[p + 1] = 0.0;
      }

      string name=lout[k]->name;

      fprintf(stdout, "%s ( ", name.c_str());
      if (losses.size()>=(k+1)) {
        fprintf(stdout, "loss[%s]=%1.3f ", losses[k]->name.c_str(), total_loss[k] / (length*inferenced_samples));
      }
      if (metrics.size()>=(k+1)) {
        fprintf(stdout, "metric[%s]=%1.3f ", metrics[k]->name.c_str(), total_metric[k] / (length*inferenced_samples));
      }

      fprintf(stdout, ") -- ");


      if ((flog_tr!=nullptr)&&(trmode)) {
        fprintf(flog_tr, "%s ", name.c_str());
        if (losses.size()>=(k+1)) {
          fprintf(flog_tr, "loss[%s]=%1.3f ", losses[k]->name.c_str(), total_loss[k] / inferenced_samples);
        }
        if (metrics.size()>=(k+1)) {
          if (metrics[k]->name!="none")
          fprintf(flog_tr, "metric[%s]=%1.3f ", metrics[k]->name.c_str(), total_metric[k] / inferenced_samples);
        }

        fprintf(flog_tr, " -- ");


      }
      if ((flog_ts!=nullptr)&&(!trmode)) {
        fprintf(flog_ts, "%s ", name.c_str());
        if (losses.size()>=(k+1)) {
          fprintf(flog_ts, "loss[%s]=%1.3f ", losses[k]->name.c_str(), total_loss[k] / inferenced_samples);
        }
        if (metrics.size()>=(k+1)) {
          if (metrics[k]->name!="none")
          fprintf(flog_ts, "metric[%s]=%1.3f ", metrics[k]->name.c_str(), total_metric[k] / inferenced_samples);
        }

        fprintf(flog_ts, " -- ");
      }


    }
    fflush(stdout);

    if ((flog_tr!=nullptr)&&(trmode)) {
      fprintf(flog_tr, "\n");
      fflush(flog_tr);
    }

    if ((flog_ts!=nullptr)&&(!trmode)) {
      fprintf(flog_ts, "\n");
      fflush(flog_ts);
    }
  }
}

vector<float> Net::get_losses(){
    // NOTE: I have no idea how the internals of the metrics/loss values work,
    // so I did a sort of copy and paste from print_loss fuction with minor workarounds
    vector<float> loss_values;

    if (this->isrecurrent) {
        if (this->rnet!=nullptr) { this->rnet->get_losses(); } // Dangerous A.F.
    } else {
        int p = 0;

        // Copy total_loss / fiterr (I don't know how it works but i don't like it...)
        vector<float> tmp_total_error(total_loss);
        vector<float> tmp_fiterr(fiterr);

        int length=decsize;
        for (int k = 0; k < lout.size(); k+=decsize) {

            // Do stuff
            for(int l=0;l<length;l++,p+=2) {
                tmp_total_error[k] += tmp_fiterr[p];  // loss
                tmp_fiterr[p] = tmp_fiterr[p + 1] = 0.0;
            }

            // Compute average loss
            if (losses.size()>=(k+1)) {
                loss_values.push_back(tmp_total_error[k] /  (float)(length*inferenced_samples));
            }
        }
    }

    return loss_values;
}

vector<float> Net::get_metrics(){
    // NOTE: I have no idea how the internals of the metrics/loss values work,
    // so I did a sort of copy and paste from print_loss fuction with minor workarounds
    vector<float> metrics_values;

    if (this->isrecurrent) {
        if (this->rnet!=nullptr) { this->rnet->get_metrics(); } // Dangerous A.F.
    } else {
        int p = 0;

        // Copy total_loss / fiterr (I don't know how it works but i don't like it...)
        vector<float> tmp_total_metrics(total_metric);
        vector<float> tmp_fiterr(fiterr);

        int length=decsize;
        for (int k = 0; k < lout.size(); k+=decsize) {

            for(int l=0;l<length;l++,p+=2) {
                tmp_total_metrics[k] += tmp_fiterr[p + 1];  // metric
                tmp_fiterr[p] = tmp_fiterr[p + 1] = 0.0;
            }

            if (metrics.size()>=(k+1)) {
                metrics_values.push_back( tmp_total_metrics[k] / (float)(length*inferenced_samples));
            }

        }
    }

    return metrics_values;
}

void Net::reset_grads()
{
  if (isrecurrent)
  if (rnet!=nullptr)
  rnet->reset_grads();

  do_reset_grads();
  run_snets(reset_grads_t);
}

void Net::reset()
{
  if (isrecurrent)
  if (rnet!=nullptr)
  rnet->reset();

  do_reset();
  run_snets(reset_t);
}




void Net::update()
{
  if (isrecurrent) {
    if (rnet!=nullptr) {
      rnet->update();
    }
  }
  else {
    run_snets(update_t);

    int comp=snets.size();

    if (batch_size<comp) {
      msg("batch_size lower than computing service parallelism","update");

    }

    if ((snets[0]->dev != DEV_CPU) && (comp > 1) && (tr_batches%cs->lsb==1)) {
      sync_weights();
    }
  }
}

void Net::delta()
{
  if (isrecurrent) {
    if (rnet!=nullptr)
    rnet->run_snets(delta_t);
  }
  else run_snets(delta_t);

}


//////////////////////////////////////////////////////////////
//////// HIGHER LEVEL FUNCS
void Net::fit(vtensor tin, vtensor tout, int batch, int epochs) {
  int i, j, k, n;

  if (isrecurrent) {
    fit_recurrent(tin,tout, batch, epochs);
  }
  else{

    // Check current optimizer
    if (optimizer == nullptr)
    msg("Net is not build", "Net.fit");

    // Check if number of input/output network layers matches with the input/output tensor data
    if (tin.size() != lin.size()) {
      cout<<tin.size()<<"!="<<lin.size()<<endl;
      msg("input tensor list does not match with defined input layers", "Net.fit");
    }
    if (tout.size() != lout.size()) {
      cout<<tout.size()<<"!="<<lout.size()<<endl;
      msg("output tensor list does not match with defined output layers", "Net.fit");
    }


    // Check if all the data inputs has the same number of samples
    n = tin[0]->shape[0];
    for (i = 1; i < tin.size(); i++)
    if (tin[i]->shape[0] != n)
    msg("different number of samples in input tensor", "Net.fit");



    // Check if the size of the output layers matches with inputs sizes
    for (i = 1; i < tout.size(); i++)
    if (tout[i]->shape[0] != n)
    msg("different number of samples in output tensor", "Net.fit");



    // Set batch size
    resize(batch);


    // Create array to store batch indices (later random)
    vind sind;
    for (i = 0; i < batch_size; i++)
    sind.push_back(0);


    // Start training
    setmode(TRMODE);

    // Set some parameters
    int num_batches = n / batch_size;

    // Train network
    fprintf(stdout, "%d epochs of %d batches of size %d\n", epochs, num_batches, batch_size);
    for (i = 0; i < epochs; i++) {
      high_resolution_clock::time_point e1 = high_resolution_clock::now();
      fprintf(stdout, "Epoch %d\n", i + 1);

      reset_loss();

      // For each batch
      for (j = 0; j < num_batches; j++) {

        // Set random indices
        for (k = 0; k < batch_size; k++) sind[k] = rand() % n;

        // Train batch
        tr_batches++;

        train_batch(tin, tout, sind);

        print_loss(j+1);

        high_resolution_clock::time_point e2 = high_resolution_clock::now();
        duration<double> epoch_time_span = e2 - e1;
        fprintf(stdout, "%1.3f secs/batch\r", epoch_time_span.count()/(j+1));
        fflush(stdout);


      }
      high_resolution_clock::time_point e2 = high_resolution_clock::now();
      duration<double> epoch_time_span = e2 - e1;
      fprintf(stdout, "\n%1.3f secs/epoch\n", epoch_time_span.count());
    }
    fflush(stdout);
  }

}


void Net::prepare_recurrent_dec(vtensor tin, vtensor tout, int &inl, int &outl, vtensor &xt, vtensor &xtd,vtensor &yt,vtensor &tinr,vtensor &toutr, Tensor *Z)
{
  int i, j, k, n;

  // Check whether is encoder, decoder or both.
  for(i=0;i<vfts.size();i++) {
    if (vfts[i]->isdecoder) {isdecoder=true;break;}
    else if (vfts[i]->isrecurrent) isencoder=true;
  }

  // Set the properties to snets
  for(i=0;i<snets.size();i++) {
    snets[i]->isdecoder=isdecoder;
    snets[i]->isencoder=isencoder;
  }

  inl=outl=1;

  for(i=0;i<tin.size();i++) {
    xt.push_back(tin[i]->clone());
    tinr.push_back(new Tensor(xt[i]->shape,xt[i]->ptr,xt[i]->device));
  }


  // PREPARE INPUT and OUTPUT DECODER
  for(i=0;i<tout.size();i++) {
    if (tout[i]->ndim<3)
      msg("Output tensor should be batch x timesteps x dims","Net::prepare_recurrent");
    vector<int> pshape;
    pshape.push_back(1);
    pshape.push_back(0);
    for(int j=2;j<tout[i]->ndim;j++)
     pshape.push_back(j);
    yt.push_back(Tensor::permute(tout[i],pshape)); // time x batch x dim
  }

  // outl=out time_steps.
  // Check that all the potential inputs have the same timesteps:
  outl=yt[0]->shape[0];
  for(i=0;i<yt.size();i++) {
    if (yt[i]->shape[0]!=outl)
    msg("Output tensors with different time steps","fit_recurrent");
  }

  int offset;
  for(i=0;i<yt.size();i++) {
    offset=yt[i]->size/yt[i]->shape[0];
    vector<int>shape;
    for(j=1;j<yt[i]->ndim;j++)
      shape.push_back(yt[i]->shape[j]);

    //input, delayed
    vector<int>zero_shape;
    for(j=0;j<tout[i]->ndim;j++)
      if (j!=1) zero_shape.push_back(tout[i]->shape[j]);

    tinr.push_back(Tensor::zeros(zero_shape,tout[i]->device));
    for(j=0;j<outl-1;j++)
      tinr.push_back(new Tensor(shape,yt[i]->ptr+(j*offset),yt[i]->device));

    // output
    for(j=0;j<outl;j++)
      toutr.push_back(new Tensor(shape,yt[i]->ptr+(j*offset),yt[i]->device));
  }



}



void Net::prepare_recurrent_enc_dec(vtensor tin, vtensor tout, int &inl, int &outl, vtensor &xt, vtensor &xtd,vtensor &yt,vtensor &tinr,vtensor &toutr, Tensor *Z)
{

  int i, j, k, n;

  // Check whether is encoder, decoder or both.
  for(i=0;i<vfts.size();i++) {
    if (vfts[i]->isdecoder) {isdecoder=true;break;}
    else if (vfts[i]->isrecurrent) isencoder=true;
  }

  // Set the properties to snets
  for(i=0;i<snets.size();i++) {
    snets[i]->isdecoder=isdecoder;
    snets[i]->isencoder=isencoder;
  }

  inl=outl=1;

  // PREPARE INPUT ENCODER
  for(i=0;i<tin.size();i++) {
    if (tin[i]->ndim<3)
        msg("Input tensor should be batch x timesteps x dims","Net::prepare_recurrent");
    vector<int> pshape;
    pshape.push_back(1);
    pshape.push_back(0);
    for(int j=2;j<tin[i]->ndim;j++)
     pshape.push_back(j);
    xt.push_back(Tensor::permute(tin[i],pshape)); // time x batch x dims
  }

  // inl=input time_steps.
  // Check that all the potential inputs have the same timesteps:
  inl=xt[0]->shape[0];
  for(i=0;i<xt.size();i++) {
    if (xt[i]->shape[0]!=inl)
      msg("Input tensors with different time steps","Net::prepare_recurrent");
  }


  int offset;
  for(i=0;i<xt.size();i++) { // again xt.size() is normally 1
    offset=xt[i]->size/xt[i]->shape[0];
    vector<int>shape;
    for(j=1;j<xt[i]->ndim;j++)
      shape.push_back(xt[i]->shape[j]);
    for(j=0;j<inl;j++) // fot all timesteps create a share tensor
      tinr.push_back(new Tensor(shape,xt[i]->ptr+(j*offset),xt[i]->device));
  }


  // PREPARE INPUT and OUTPUT DECODER
  for(i=0;i<tout.size();i++) {
    if (tout[i]->ndim<3)
      msg("Output tensor should be batch x timesteps x dims","Net::prepare_recurrent");
    vector<int> pshape;
    pshape.push_back(1);
    pshape.push_back(0);
    for(int j=2;j<tout[i]->ndim;j++)
     pshape.push_back(j);
    yt.push_back(Tensor::permute(tout[i],pshape)); // time x batch x dim
  }

  // outl=out time_steps.
  // Check that all the potential inputs have the same timesteps:
  outl=yt[0]->shape[0];
  for(i=0;i<yt.size();i++) {
    if (yt[i]->shape[0]!=outl)
    msg("Output tensors with different time steps","fit_recurrent");
  }

  for(i=0;i<yt.size();i++) {
    offset=yt[i]->size/yt[i]->shape[0];
    vector<int>shape;
    for(j=1;j<yt[i]->ndim;j++)
      shape.push_back(yt[i]->shape[j]);

    //input, delayed
    vector<int>zero_shape;
    for(j=0;j<tout[i]->ndim;j++)
      if (j!=1) zero_shape.push_back(tout[i]->shape[j]);

    tinr.push_back(Tensor::zeros(zero_shape,tout[i]->device));
    for(j=0;j<outl-1;j++)
      tinr.push_back(new Tensor(shape,yt[i]->ptr+(j*offset),yt[i]->device));

    // output
    for(j=0;j<outl;j++)
      toutr.push_back(new Tensor(shape,yt[i]->ptr+(j*offset),yt[i]->device));
  }




}


void Net::prepare_recurrent_enc(vtensor tin, vtensor tout, int &inl, int &outl, vtensor &xt, vtensor &xtd,vtensor &yt,vtensor &tinr,vtensor &toutr, Tensor *Z)
{
  int i, j, k, n;

  // Check whether is encoder, decoder or both.
  for(i=0;i<vfts.size();i++) {
    if (vfts[i]->isdecoder) {isdecoder=true;break;}
    else if (vfts[i]->isrecurrent) isencoder=true;
  }

  // Set the properties to snets
  for(i=0;i<snets.size();i++) {
    snets[i]->isdecoder=isdecoder;
    snets[i]->isencoder=isencoder;
  }

  inl=outl=1;

  // PREPARE INPUT
  for(i=0;i<tin.size();i++) {
    if (tin[i]->ndim<3)
        msg("Input tensor should be batch x timesteps x dims","Net::prepare_recurrent");
    vector<int> pshape;
    pshape.push_back(1);
    pshape.push_back(0);
    for(int j=2;j<tin[i]->ndim;j++)
     pshape.push_back(j);
    xt.push_back(Tensor::permute(tin[i],pshape)); // time x batch x dims

  inl=xt[0]->shape[0];
  // inl=time_steps.
  // Check that all the potential inputs have the same timesteps:
  for(i=0;i<xt.size();i++) {
    if (xt[i]->shape[0]!=inl)
      msg("Input tensors with different time steps","Net::prepare_recurrent");
    }
  }

  int offset;
  for(i=0;i<xt.size();i++) { // again xt.size() is normally 1
    offset=xt[i]->size/xt[i]->shape[0];
    vector<int>shape;
    for(j=1;j<xt[i]->ndim;j++)
      shape.push_back(xt[i]->shape[j]);
    for(j=0;j<inl;j++) // fot all timesteps create a share tensor
      tinr.push_back(new Tensor(shape,xt[i]->ptr+(j*offset),xt[i]->device));
  }

  // PREPARE OUTPUT
  for(i=0;i<tout.size();i++) {
    if (tout[i]->ndim<3)
      msg("Output tensor should be batch x timesteps x dims","Net::prepare_recurrent");
    vector<int> pshape;
    pshape.push_back(1);
    pshape.push_back(0);
    for(int j=2;j<tout[i]->ndim;j++)
     pshape.push_back(j);
    yt.push_back(Tensor::permute(tout[i],pshape)); // time x batch x dims

    outl=yt[0]->shape[0];
    // outl=time_steps output
    // Check that all the potential outputs have the same timesteps:
    for(i=0;i<yt.size();i++) {
      if (yt[i]->shape[0]!=outl)
      msg("Output tensors with different time steps","Net::prepare_recurrent");
    }
  }

  for(i=0;i<yt.size();i++) {
    offset=yt[i]->size/yt[i]->shape[0];
    vector<int>shape;
    for(j=1;j<yt[i]->ndim;j++)
      shape.push_back(yt[i]->shape[j]);

    if (tout.size())
      for(j=0;j<outl;j++)
        toutr.push_back(new Tensor(shape,yt[i]->ptr+(j*offset),yt[i]->device));
  }
}



void Net::prepare_recurrent(vtensor tin, vtensor tout, int &inl, int &outl, vtensor &xt, vtensor &xtd,vtensor &yt,vtensor &tinr,vtensor &toutr, Tensor *Z)
{
  int i, j, k, n;

  // Check whether is encoder, decoder or both.
  for(i=0;i<vfts.size();i++) {
    if (vfts[i]->isdecoder) {isdecoder=true;break;}
    else if (vfts[i]->isrecurrent) isencoder=true;
  }

  // Set the properties to snets
  for(i=0;i<snets.size();i++) {
    snets[i]->isdecoder=isdecoder;
    snets[i]->isencoder=isencoder;
  }

  if ((isencoder)&&(isdecoder))
    prepare_recurrent_enc_dec(tin, tout, inl, outl, xt, xtd, yt, tinr,toutr);
  else if (isdecoder)
      prepare_recurrent_dec(tin, tout, inl, outl, xt, xtd, yt, tinr,toutr);
  else
    prepare_recurrent_enc(tin, tout, inl, outl, xt, xtd, yt, tinr,toutr);
}

void Net::fit_recurrent(vtensor tin, vtensor tout, int batch, int epochs) {
  int i, j, k, n;

  // prepare data for unroll net
  vtensor xt;
  vtensor xtd;
  vtensor yt;

  vtensor toutr;
  vtensor tinr;

  int inl;
  int outl;

  prepare_recurrent(tin,tout,inl,outl,xt,xtd,yt,tinr,toutr);

  build_rnet(inl,outl);

  rnet->fit(tinr,toutr,batch,epochs);

  if (snets[0]->dev!=DEV_CPU) rnet->sync_weights();

  for(i=0;i<tinr.size();i++) delete(tinr[i]);
  for(i=0;i<toutr.size();i++) delete(toutr[i]);


  for(i=0;i<xt.size();i++)
    delete xt[i];
  xt.clear();

  for(i=0;i<yt.size();i++)
    delete yt[i];
  yt.clear();

}

// TODO:  train_batch_recurrent
/////////////////////////////////////////
void Net::train_batch(vtensor X, vtensor Y, vind sind, int eval) {

  if (batch_size!=sind.size()) resize(sind.size());

  int comp=snets.size();

  if (batch_size<comp) {
    msg("batch_size lower than computing service parallelism","compute_loss");
  }

  int thread_batch_size=batch_size / comp;

  if (eval) setmode(TSMODE);
  else setmode(TRMODE);

  // Check indices
  if (sind.size() == 0) msg("error void index","Net::train_batch");
  // Split data for each network
  for (int i = 0; i < comp; i++) {
    int start = i * thread_batch_size;
    int end = start + Xs[i][0]->shape[0];

    // Copy samples
    for (int j = 0; j < X.size(); j++) {
      Tensor::select(X[j], Xs[i][j], sind, start, end);
      Tensor::copy(Xs[i][j], snets[i]->lin[j]->input);
    }

    // Copy targets
    for (int j = 0; j < Y.size(); j++) {
      Tensor::select(Y[j], Ys[i][j], sind, start, end);
      snets[i]->lout[j]->check_target();
      Tensor::copy(Ys[i][j], snets[i]->lout[j]->target);

      /* Better do n-best
      if (isdecoder) {
        if (eval) {
          if (j==0) snets[i]->din[0]->input->fill_(0.0); //start
          else {
            snets[i]->lout[j-1]->addchild(snets[i]->din[j]);
          }
        }
        else {
          if (j==0) snets[i]->din[0]->input->fill_(0.0); //start
          else Tensor::copy(Ys[i][j-1], snets[i]->din[j]->input);
        }
      }
      */
    }
  }

  if (eval)
  run_snets(eval_batch_t);
  else
  run_snets(train_batch_t);

  /*
  if ((eval)&&(isdecoder))
    for (int i = 0; i < comp; i++)
      for (int j = 1; j < Y.size(); j++)
         snets[i]->lout[j-1]->detach(snets[i]->din[j]);
  */

  // If training (eval==0), apply gradients
  if (!eval) {
    // In case of multiple GPUS or FPGA synchronize params
    if ((snets[0]->dev != DEV_CPU) && (comp > 1) && (tr_batches%cs->lsb==0)) {
      sync_weights();
    }
  }

  compute_loss();

#ifdef cFPGA
  _show_profile_fpga();
#endif

}


///////////////////////////////////////////
void Net::evaluate(vtensor tin, vtensor tout,int bs) {

  int i, j, k, n;

  if (isrecurrent) {
    evaluate_recurrent(tin,tout,bs);
  }
  else{

    // Check list shape
    if (tin.size() != lin.size())
    msg("input tensor list does not match with defined input layers", "Net.evaluate");
    if (tout.size() != lout.size())
    msg("output tensor list does not match with defined output layers", "Net.evaluate");

    // Check data consistency
    n = tin[0]->shape[0];

    for (i = 1; i < tin.size(); i++)
    if (tin[i]->shape[0] != n)
    msg("different number of samples in input tensor", "Net.evaluate");

    for (i = 1; i < tout.size(); i++)
    if (tout[i]->shape[0] != n)
    msg("different number of samples in output tensor", "Net.evaluate");

    resize(bs);

    printf("Evaluate with batch size %d\n",batch_size);

    // Create internal variables
    vind sind;
    for (k=0;k<batch_size;k++)
    sind.push_back(0);


    // Start eval
    setmode(TSMODE);
    reset_loss();
    for (j = 0; j < n / batch_size; j++) {

      for (k=0;k<batch_size;k++)
        sind [k]=(j*batch_size)+k;

      train_batch(tin, tout, sind, 1);

      print_loss(j+1);
      fprintf(stdout, "\r");
      fflush(stdout);
    }
    fprintf(stdout, "\n");

  }
}

///////////////////////////////////////////
void Net::evaluate_recurrent(vtensor tin, vtensor tout, int bs) {

  int i;

  // prepare data for unroll net
  vtensor xt;
  vtensor xtd;
  vtensor yt;

  vtensor toutr;
  vtensor tinr;

  int inl;
  int outl;

  prepare_recurrent(tin,tout,inl,outl,xt,xtd,yt,tinr,toutr);

  build_rnet(inl,outl);

  if ((isencoder)&&(isdecoder))
    rnet->evaluate(tinr,toutr,bs);
  else if (isencoder){
    rnet->evaluate(tinr,toutr,bs);}
  else if (isdecoder)
    rnet->evaluate(tinr,toutr,bs);

  for(i=0;i<tinr.size();i++) delete(tinr[i]);
  for(i=0;i<toutr.size();i++) delete(toutr[i]);

  if (isencoder) {
    for(i=0;i<xt.size();i++)
      delete xt[i];
    xt.clear();
  }
  if (isdecoder) {
    for(i=0;i<xtd.size();i++)
      delete xtd[i];
    xtd.clear();

    for(i=0;i<yt.size();i++)
      delete yt[i];
    yt.clear();
  }

}



///////////////////////////////////////////
vtensor Net::predict_recurrent(vtensor tin) {
  vtensor out;

  // prepare data for unroll net
  vtensor xt;
  vtensor xtd;
  vtensor yt;

  vtensor toutr;
  vtensor tinr;
  vtensor tout;

  int inl;
  int outl;

  prepare_recurrent(tin,tout,inl,outl,xt,xtd,yt,tinr,toutr);

  out=rnet->predict(tinr);

  for(int i=0;i<xt.size();i++)
    delete xt[i];
  xt.clear();

  return out;
}

vtensor Net::predict(vtensor tin) {
  vtensor out;

  if (isrecurrent) {
    return predict_recurrent(tin);
  }
  else {
    cout<<"Predict "<<tin[0]->shape[0]<<" samples\n";

    setmode(TSMODE);

    forward(tin);

    for (int i = 0; i < lout.size(); i++) {
      collectTensor(lout[i],"output");
      out.push_back(lout[i]->output->clone());
    }
    return out;
  }

}

//////
