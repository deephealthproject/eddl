
/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_NET_H
#define EDDL_NET_H

#include <string>
#include <vector>

#include "../layers/layer.h"
#include "../optimizers/optim.h"
#include "../losses/loss.h"
#include "../metrics/metric.h"
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


#define MAX_THREADS 1024

class Net {
private:
    void build(Optimizer *opt, vloss lo, vmetrics me);

    void set_compserv(CompServ *cs);

public:
    string name;
    int dev;
    int batch_size;
    int tr_batches;
    int inferenced_samples;
    int trmode;

    vector<int> devsel;
    CompServ *cs;

    vlayer layers;
    vlayer lin;
    vlayer lout;
    vlayer vfts;
    vlayer vbts;

    vloss losses;
    vmetrics metrics;
    verr fiterr;
    verr total_loss;
    verr total_metric;
    FILE *flog_tr;
    FILE *flog_ts;

    Optimizer *optimizer;
    vector<Net *> snets;

    vtensor Xs[MAX_THREADS];
    vtensor Ys[MAX_THREADS];

    Net(vlayer in, vlayer out);
    ~Net();

    void initialize();
    void reset();
    void save(FILE *fe);
    void load(FILE *fe);
    void setlogfile(string fname);


    void forward();
    void delta();
    void calcloss();
    void backward();
    void applygrads();

    void split(int c, int todev);
    int inNet(Layer *l); //
    void walk(Layer *l); //
    void walk_back(Layer *l); //

    void fts();
    void bts();

    void resize(int batch);

    string summary();
    void plot(string fname,string mode);

    void setmode(int m);
    void sync_weights();

    Layer *getLayer(string name);

    void build(Optimizer *opt, vloss lo, vmetrics me, CompServ *cs); //

    void fit(vtensor tin, vtensor tout, int batch_size, int epochs);

    void train_batch(vtensor X, vtensor Y, vind sind, int eval = 0);

    void evaluate(vtensor tin, vtensor tout);
    void predict(vtensor tin, vtensor tout);

    void reset_loss();
    void print_loss(int b);

    void forward(vector<Tensor *> in);
    void backward(vector<Tensor *> target);
    void backward(Layer* (*f)(Layer *),Layer *out);
    void reset_grads();
    void update();
    void compute_loss();


};

#endif  //EDDL_NET_H
