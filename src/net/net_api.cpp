/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include "net.h"
#include <pthread.h>
#include "../utils.h"
#include "../random.h"
#include "../layers/core/layer_core.h"

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
  pthread_t thr[100];
  struct tdata td[100];

  int comp=snets.size();

  if (batch_size<comp)
  comp=batch_size;

  for (int i = 0; i < comp; i++) {
    // Thread params
    td[i].net = snets[i];

    rc = pthread_create(&thr[i], nullptr, (*F), (void *) (&td[i]));
    if (rc) {
        fprintf(stderr, "Error:unable to create thread %d", rc);
        exit(-1);
    }
  }

  // Wait until all threads have finished
  for (int i = 0; i < comp; i++) {
    rc = pthread_join(thr[i], &status);
    if (rc) {
        cout << "Error:unable to join," << rc << endl;
        exit(-1);
    }
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

//////////////////////////////////
// API functions

void Net::forward(vector<Tensor*> in)
{

  netinput.clear();

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

    for (int i = 0; i < in.size(); i++)
      copyTensor(in[i],lin[i]);

  }

  run_snets(forward_t);

}

void Net::forward()
{
  reset();

  run_snets(forward_t);
}


void Net::backward(vector<Tensor *> target)
{
  void *status;
  int rc;
  pthread_t thr[100];
  struct tdata td[100];


  int comp=snets.size();

  if (target.size()) {
    if (target.size()!=lout.size())
      msg("size missmatch in list of targets","Net.backward(vtensor)");

    if (batch_size!=target[0]->shape[0])
      msg("bakcward step with different batch_size than forward","Net.backward(vtensor)");


    if (batch_size<comp)
      comp=batch_size;

    int thread_batch_size=batch_size / comp;

    // Split data for each network
    for (int i = 0; i < comp; i++) {
        int start = i * thread_batch_size;
        int end = start + Ys[i][0]->shape[0];
        vector<int> sind(batch_size);
        for(int k=0;k<batch_size;k++) sind[k]=k;
        // Copy samples
        // Copy targets
        for (int j = 0; j < target.size(); j++) {
            Tensor::select(target[j], Ys[i][j], sind, start, end);
            Tensor::copy(Ys[i][j], snets[i]->lout[j]->target);
        }
    }
  }
  tr_batches++;
  run_snets(backward_t);
  compute_loss();

}


void Net::backward()
{

  tr_batches++;
  run_snets(backward_t);

  for(int i=0;i<netinput.size();i++) {
    if (netinput[i]->detached==false) {
      copyTensor(lin[i],netinput[i],"grad");
      netinput[i]->net->backward();
    }
  }

}

void Net::reset_loss()
{
  // Reset errors
  int p=0;
  for (int j = 0; j < lout.size(); j++,p+=2){
      total_loss[j] = 0.0;
      total_metric[j] = 0.0;
      fiterr[p] = fiterr[p + 1] = 0.0;
  }
  inferenced_samples=0;
}

void Net::print_loss(int b)
{
  int p = 0;

  for (int k = 0; k < lout.size(); k++, p += 2) {
      total_loss[k] += fiterr[p];  // loss
      total_metric[k] += fiterr[p + 1];  // metric
      fiterr[p] = fiterr[p + 1] = 0.0;
      fprintf(stdout,"Batch %d ",b);
      fprintf(stdout, "%s(%s=%1.3f,%s=%1.3f) ", lout[k]->name.c_str(),
              losses[k]->name.c_str(), total_loss[k] / inferenced_samples,
              metrics[k]->name.c_str(), total_metric[k] / inferenced_samples);

      if ((flog_tr!=nullptr)&&(trmode))
        fprintf(flog_tr, "%s %1.3f %s %1.3f ", losses[k]->name.c_str(), total_loss[k] / inferenced_samples,
                metrics[k]->name.c_str(), total_metric[k] / inferenced_samples);

      if ((flog_ts!=nullptr)&&(!trmode))
        fprintf(flog_ts, "%s %1.3f %s %1.3f ", losses[k]->name.c_str(), total_loss[k] / inferenced_samples,
                metrics[k]->name.c_str(), total_metric[k] / inferenced_samples);

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

void Net::reset_grads()
{
  do_reset_grads();
  run_snets(reset_grads_t);
}

void Net::reset()
{
  do_reset();
  run_snets(reset_t);
}

void Net::compute_loss()
{
  run_snets(compute_loss_t);

  int comp=snets.size();
  if (batch_size<comp)
    comp=batch_size;

  if (snets[0]->dev != DEV_CPU)
    for (int i = 0; i < comp; i++) {
        for (int j = 0; j < 2 * lout.size(); j++) {
            fiterr[j] += snets[i]->fiterr[j];
        }
    }

    inferenced_samples+=batch_size;
}


void Net::update()
{
  run_snets(update_t);

  int comp=snets.size();

  if (batch_size<comp)
    comp=batch_size;

  if ((snets[0]->dev != DEV_CPU) && (comp > 1) && (tr_batches%cs->lsb==1)) {
    sync_weights();
  }
}

void Net::delta()
{
  run_snets(update_t);
}


//////////////////////////////////////////////////////////////
//////// HIGHER LEVEL FUNCS
void Net::fit(vtensor tin, vtensor tout, int batch, int epochs) {
    int i, j, k, n;

    // Check current optimizer
    if (optimizer == nullptr)
        msg("Net is not build", "Net.fit");

    // Check if number of input/output network layers matches with the input/output tensor data
    if (tin.size() != lin.size())
        msg("input tensor list does not match with defined input layers", "Net.fit");
    if (tout.size() != lout.size())
        msg("output tensor list does not match with defined output layers", "Net.fit");

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

            print_loss(j);

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

/////////////////////////////////////////
void Net::train_batch(vtensor X, vtensor Y, vind sind, int eval) {
  void *status;
  int rc;
  pthread_t thr[100];
  struct tdata td[100];


  if (batch_size!=sind.size()) resize(sind.size());

  int comp=snets.size();

  if (batch_size<comp)
    comp=batch_size;

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
        Tensor::copy(Ys[i][j], snets[i]->lout[j]->target);
    }
  }

  if (eval)
    run_snets(eval_batch_t);
  else
    run_snets(train_batch_t);

  // If training (eval==0), apply gradients
  if (!eval) {
    // In case of multiple GPUS or FPGA synchronize params
    if ((snets[0]->dev != DEV_CPU) && (comp > 1) && (tr_batches%cs->lsb==0)) {
      sync_weights();
    }
  }

  compute_loss();

}


///////////////////////////////////////////
void Net::evaluate(vtensor tin, vtensor tout) {

    int i, j, k, n;

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
          sind[k]=(j*batch_size)+k;

        train_batch(tin, tout, sind, 1);

        print_loss(j);
        fprintf(stdout, "\r");
        fflush(stdout);
    }
    fprintf(stdout, "\n");

}
















//////
