/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
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
    int eval;
};



/////////////////////////////////////////
void *train_batch_t(void *t) {
    auto *targs = (tdata *) t;

    Net *net = targs->net;
    net->reset();
    net->forward();
    net->calcloss();

    if (!targs->eval) {
        net->delta();
        net->backward();
        if (net->dev > DEV_CPU)
            net->applygrads();
    }

    return nullptr;
}
/////////////////////////////////////////
void *forward_t(void *t) {
    auto *targs = (tdata *) t;

    Net *net = targs->net;

    net->forward();

    return nullptr;
}

/////////////////////////////////////////
void *reset_t(void *t) {
    auto *targs = (tdata *) t;

    Net *net = targs->net;

    net->reset();

    return nullptr;
}

/////////////////////////////////////////
void *backward_t(void *t) {
    auto *targs = (tdata *) t;

    Net *net = targs->net;

    net->delta();
    net->backward();

    return nullptr;
}

void *calcloss_t(void *t)
{
  auto *targs = (tdata *) t;

  Net *net = targs->net;

  net->calcloss();

  return nullptr;
}

/////////////////////////////////////////
void *update_t(void *t) {
    auto *targs = (tdata *) t;

    Net *net = targs->net;
    net->applygrads();

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
    td[i].eval = 0;

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


void Net::forward(vector<Tensor *> in)
{
  void *status;
  int rc;
  pthread_t thr[100];
  struct tdata td[100];

  int comp=snets.size();

  if (in.size()) {
    if (in.size()!=lin.size())
      msg("size missmatch in list of tensors","Net.forward(vtensor)");

    if (batch_size!=in[0]->shape[0]) {
      resize(in[0]->shape[0]);
    }

    if (batch_size<comp)
      comp=batch_size;

    int thread_batch_size=batch_size / comp;

    // Split data for each network
    for (int i = 0; i < comp; i++) {
        int start = i * thread_batch_size;
        int end = start + Xs[i][0]->shape[0];
        vector<int> sind(batch_size);
        for(int k=0;k<batch_size;k++) sind[k]=k;

        // Copy samples
          for (int j = 0; j < in.size(); j++) {
            Tensor::select(in[j], Xs[i][j], sind, start, end);
            Tensor::copy(Xs[i][j], snets[i]->lin[j]->input);
        }
    }
  }

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
        int end = start + Xs[i][0]->shape[0];
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

  run_snets(backward_t);
}

void Net::backward(Layer* (*f)(Layer *),Layer *out)
{
  int comp=snets.size();
  if (batch_size<comp)
    comp=batch_size;


  Layer *input=new LInput(new Tensor(out->output->getShape()),"lossnet_input",DEV_CPU);
  Layer *fout=(*f)(input);

  Net *lossnet=new Net({input},{fout});

  lossnet->build(optimizer,{new LMin()},{new MSum()},cs);

  Net *n=out->net;
  Net *sn=out->net->snets[0];

  int i;
  Layer *sout=nullptr;
  for(i=0;i<sn->layers.size();i++)
   if (sn->layers[i]->orig==out) {
      sout=sn->layers[i];
      break;
   }
  if (sout==nullptr)
    msg("layer not found in subgrap","Net::backward(loss_func)");

  sout->output->print();
  sout->output->info();

  lossnet->reset_loss();
  lossnet->forward({sout->output});

  lossnet->reset_grads();
  lossnet->backward({});
  lossnet->compute_loss();

  lossnet->print_loss(1);

  //Tensor::copy(input->delta,sout->delta);
  //sn->backward({});

  getchar();

  delete lossnet;

  // copy delta to out and call backward of orig net...

}


void Net::reset_grads()
{
  run_snets(reset_t);
}

void Net::compute_loss()
{
  run_snets(calcloss_t);

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

    run_snets(train_batch_t);

    // If training (eval==0), apply gradients
    if (!eval) {
        if (snets[0]->dev == DEV_CPU) {
            snets[0]->applygrads();
        }
        // In case of multiple GPUS or FPGA synchronize params
        if ((snets[0]->dev != DEV_CPU) && (comp > 1) && (tr_batches%cs->lsb==0)) {
          sync_weights();
        }
    }

   compute_loss();

}


/////////////////////////////////////////
void Net::sync_weights() {
    for (int j = 0; j < layers.size(); j++)
        for (int k = 0; k < layers[j]->params.size(); k++) {
            // Taking average
            layers[j]->params[k]->fill_(0.0);
            for (int i = 0; i < snets.size(); i++) {
                Tensor::inc(snets[i]->layers[j]->params[k], layers[j]->params[k]);
            }
            layers[j]->params[k]->div_(snets.size());

            // copy-back to devices
            for (int i = 0; i < snets.size(); i++) {
                Tensor::copy(layers[j]->params[k], snets[i]->layers[j]->params[k]);
            }
        }
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


    if (n<batch_size) resize(n);

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

///////////////////////////////////////////
void Net::predict(vtensor tin, vtensor tout) {

    int i, j, k, n;
    setmode(TSMODE);

    // Check list shape
    if (tin.size() != lin.size())
        msg("input tensor list does not match with defined input layers", "Net.predict");
    if (tout.size() != lout.size())
        msg("output tensor list does not match with defined output layers", "Net.predict");

    // Check data consistency
    n = tin[0]->shape[0];
    if (n!=1)
      msg("Predict only one sample","Net.predict");

    for (i = 1; i < tin.size(); i++)
        if (tin[i]->shape[0] != n)
            msg("different number of samples in input tensor", "Net.predict");

    for (i = 1; i < tout.size(); i++)
        if (tout[i]->shape[0] != n)
            msg("different number of samples in output tensor", "Net.predict");


    if (batch_size!=1) resize(1);

    printf("Predict...\n");

    // Copy samples
    for (int j = 0; j < tin.size(); j++)
        Tensor::copy(tin[j], snets[0]->lin[j]->input);

    snets[0]->reset();
    snets[0]->forward();

    for (int j = 0; j < tout.size(); j++) {
        Tensor::copy(snets[0]->lout[j]->output,tout[j]);
    }

}


















//////
