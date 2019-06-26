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


#ifndef EDDLL_NET_H
#define EDDLL_NET_H

#include <stdio.h>
#include <string>
#include <initializer_list>
#include <vector>

#include "layers/layer.h"
#include "optimizers/optim.h"
#include "losses/loss.h"
#include "metrics/metric.h"
#include "compserv.h"

using namespace std;

typedef vector<Layer *> vlayer;
typedef vector<Tensor *> vtensor;
typedef vector<vtensor> Mtensor;
typedef vector<string> vstring;
typedef vector<float> verr;
typedef vector<int> vind;
typedef vector<Loss *> vloss;
typedef vector<Metric *> vmetrics;

void *train_batch_t(void *targs);

void *applygrads_t(void *t);

#define MAX_THREADS 1024

class Net {
private:
    void train_batch(vtensor X, vtensor Y, vind sind, int eval = 0);

    void build(Optimizer *opt, vloss lo, vmetrics me);

    void set_compserv(CompServ *cs);

public:
    string name;
    int dev;
    int batch_size;
    vector<int> devsel;

    vlayer layers;
    vlayer lin;
    vlayer lout;
    vlayer vfts;
    vlayer vbts;

    vloss losses;
    vmetrics metrics;
    verr fiterr;

    Optimizer *optimizer;
    vector<Net *> snets;

    vtensor Xs[MAX_THREADS];
    vtensor Ys[MAX_THREADS];

    Net(vlayer in, vlayer out);

    void initialize();
    void reset();
    void forward();
    void delta();
    void loss();
    void backward();
    void applygrads();

    void split(int c, int todev);
    int inNet(Layer *l); //
    void walk(Layer *l); //

    void fts();
    void bts();

    void resize(int batch);

    string summary();
    void plot(string fname);

    void setmode(int m);
    void sync_weights();
    void clean_fiterr();


    Layer *getLayer(string name);

    void build(Optimizer *opt, vloss lo, vmetrics me, CompServ *cs); //

    void fit(vtensor tin, vtensor tout, int batch_size, int epochs);

    void train_batch_ni(vector<Tensor *> in, vector<Tensor *> out);

    void evaluate(vtensor tin, vtensor tout);

};

#endif  //EDDLL_NET_H
