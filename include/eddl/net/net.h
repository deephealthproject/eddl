
/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_NET_H
#define EDDL_NET_H

#include <string>
#include <vector>

#include "eddl/layers/layer.h"
#include "eddl/optimizers/optim.h"
#include "eddl/losses/loss.h"
#include "eddl/metrics/metric.h"
#include "eddl/net/compserv.h"

using namespace std;

typedef vector<Layer *> vlayer;
typedef vector<Tensor *> vtensor;
typedef vector<vtensor> Mtensor;
typedef vector<string> vstring;
typedef vector<float> verr;
typedef vector<int> vind;
typedef vector<Loss *> vloss;
typedef vector<Metric *> vmetrics;


/////////////////////////////////////////
int isIn(Layer *l, vlayer vl, int &ind);
int isInorig(Layer *l, vlayer vl, int &ind);

#define MAX_THREADS 1024

class Net {
private:
    void make_graph(Optimizer *opt, vloss lo, vmetrics me, bool initialize=true);

    void check_compserv_compatibility(CompServ *cs);

    void set_compserv(CompServ *cs, bool do_compserv_delete);

public:
    string name;
    int dev;
    int batch_size;
    int tr_batches;
    int inferenced_samples;
    int trmode;
    int mem_level; // see Computing Service
    unsigned int verbosity_level = 0;
    bool onnx_pretrained;
    bool isrecurrent;
    bool isbuild;
    bool isdecoder;
    bool isencoder;
    bool isresized;
    bool decoder_teacher_training;
    int decsize;
    int quantization_training;

    vector<int> devsel;
    CompServ *cs;
    bool do_compserv_delete;

    vlayer layers;
    vlayer layersf;
    vlayer layersb;
    vlayer lin;
    vlayer din;
    vlayer lout;
    vlayer vfts;
    vlayer vbts;
    vlayer netinput;

    vloss losses;
    vmetrics metrics;
    verr fiterr;
    verr total_loss;
    verr total_metric;
    FILE *flog_tr;
    bool has_to_close_flog_tr;
    FILE *flog_ts;
    bool has_to_close_flog_ts;

    Optimizer *optimizer;
    bool do_optimizer_delete;
    vector<Net *> snets;
    Net* rnet;

    vtensor Xs[MAX_THREADS];
    vtensor Ys[MAX_THREADS];

    Net();
    Net(vlayer in, vlayer out);
    ~Net();


    void build(Optimizer *opt, vloss lo, vmetrics me, CompServ *cs,
               bool initialize = true,
               bool do_optimizer_delete = true,
               bool do_compserv_delete = false);
    void toGPU(vector<int> g,int lsb,int mem);
    void toCPU(int t);

    void fts();
    void bts();
    void split(int c, int todev);
    Net *unroll(int inl, int outl);
    Net *unroll_enc(int inl, int outl);
    Net *unroll_enc_dec(int inl, int outl);
    Net *unroll_dec(int inl, int outl);
    void build_rnet(int inl,int outl);
    Layer* getLayer(string l);
    void removeLayer(string l);
    void initializeLayer(string l);
    void setTrainable(string lanme, bool val);


    int inNet(Layer *l);
    int inNetF(Layer *l);
    int inNetB(Layer *l);
    void walk(Layer *l,vlayer lout);
    void walk_back(Layer *l);


    void resize(int batch);

    void enable_distributed();

    string summary(bool print_stdout=true);
    void plot(const string& fname="model.pdf", const string& rankdir="LR");

    void setmode(int m);
    void set_quantization_mode(int m);


    void save(const string& filename, const string& format="");
    void load(const string& filename, const string& format="");
    void setlogfile(const string& fname);


    //Func
    void do_initialize();
    void do_reset();
    void do_reset_grads();
    void do_forward();
    void do_delta();
    void do_compute_loss();
    void do_backward();
    void do_applygrads();

    void reset_accumulated_gradients();
    void apply_accumulated_gradients();

    void collect_acc_grads();
    void distribute_weights();
    void sync_weights();

    // API
    void run_snets(void *(*F)(void *t));
    void forward(vector<Layer *> in);
    void forward(vector<Tensor*> in);
    void forward();
    void forward_recurrent(vector<Tensor*> tin);
    void reset_loss();
    void print_loss(int b,int nb=-1);
    void backward(vector<Tensor *> target);
    void backward(Layer* (*f)(Layer *),Layer *out);
    void backward();
    void backward_recurrent(vector<Tensor *> target);
    void delta();
    void reset();
    void reset_grads();
    void update();
    void compute_loss();
    void clamp(float min,float max);
    void setlr(vector <float> p);
    vector<vtensor> get_parameters(bool deepcopy=false);
    void set_parameters(const vector<vtensor>& new_params);

    vector<float> get_losses();
    vector<float> get_metrics();

    void fit(vtensor tin, vtensor tout, int batch_size, int epochs);
    void prepare_recurrent(vtensor tin, vtensor tout, int &inl, int &outl, vtensor &xt,vtensor &xtd,vtensor &yt,vtensor &tinr,vtensor &toutr, Tensor *Z=nullptr);
    void prepare_recurrent_enc(vtensor tin, vtensor tout, int &inl, int &outl, vtensor &xt,vtensor &xtd,vtensor &yt,vtensor &tinr,vtensor &toutr, Tensor *Z=nullptr);
    void prepare_recurrent_dec(vtensor tin, vtensor tout, int &inl, int &outl, vtensor &xt,vtensor &xtd,vtensor &yt,vtensor &tinr,vtensor &toutr, Tensor *Z=nullptr);
    void prepare_recurrent_enc_dec(vtensor tin, vtensor tout, int &inl, int &outl, vtensor &xt,vtensor &xtd,vtensor &yt,vtensor &tinr,vtensor &toutr, Tensor *Z=nullptr);

    void fit_recurrent(vtensor tin, vtensor tout, int batch_size, int epochs);
    void train_batch(vtensor X, vtensor Y, vind sind, int eval = 0);
    void train_batch_recurrent(vtensor X, vtensor Y, vind sind, int eval = 0);
    void evaluate(vtensor tin, vtensor tout, int bs=100);
    void evaluate_recurrent(vtensor tin, vtensor tout, int bs);
    vtensor predict_recurrent(vtensor tin);
    vtensor predict(vtensor tin);

    void end_quantization_net();

    // Debug
    static bool compare_outputs(Net* net1, Net* net2, bool verbose=false, float atol=1e-05f, float rtol=0.0f, bool equal_nan=false);
    static bool compare_params(Net* net1, Net* net2, bool verbose=false, float atol=1e-05f, float rtol=0.0f, bool equal_nan=false);
};


void collectTensor(Layer *l,string tname="output",int p=0);
void distributeTensor(Layer *l,string tname="output", int p=0);
void quantizeLayer(Layer *l);


#endif  //EDDL_NET_H
