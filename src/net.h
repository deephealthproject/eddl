
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


#ifndef EDDL_NET_H
#define EDDL_NET_H

#include <stdio.h>
#include <string>
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
    int tr_batches;
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

    Optimizer *optimizer;
    vector<Net *> snets;

    vtensor Xs[MAX_THREADS];
    vtensor Ys[MAX_THREADS];

    Net(vlayer in, vlayer out);

    void initialize();
    void reset();
    void save(FILE *fe);
    void load(FILE *fe);


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
    void predict(vtensor tin, vtensor tout);

};

#endif  //EDDL_NET_H
