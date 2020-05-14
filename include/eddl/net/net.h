
/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
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
	void build(Optimizer *opt, vloss lo, vmetrics me, bool initialize=true);

	void set_compserv(CompServ *cs);

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

	vector<int> devsel;
	CompServ *cs;

	vlayer layers;
	vlayer lin;
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
	FILE *flog_ts;

	Optimizer *optimizer;
	vector<Net *> snets;
	vector<Net *> mnets;
	Net* rnet;

	vtensor Xs[MAX_THREADS];
	vtensor Ys[MAX_THREADS];

  Net();
	Net(vlayer in, vlayer out);
	Net(vector <Net *> vnets);
	~Net();

	void build(Optimizer *opt, vloss lo, vmetrics me, CompServ *cs, bool initialize=true);
	void toGPU(vector<int> g,int lsb,int mem);
	void toCPU(int t);

	void fts();
	void bts();
	void split(int c, int todev);
	Net *unroll(int inl, int outl, bool seq, bool areg);
	void build_rnet(int inl,int outl);

	int inNet(Layer *l);
	void walk(Layer *l);
	void walk_back(Layer *l);


	void resize(int batch);

	void enable_distributed();

	string summary();
	void plot(string fname,string mode);

	void setmode(int m);


	void save(const string& filename, string format="");
	void load(const string& filename, string format="");
	void setlogfile(string fname);


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

	void sync_weights();

	// API
	void run_snets(void *(*F)(void *t));
	void forward(vector<Layer *> in);
	void forward(vector<Tensor*> in);
	void forward();
	void forward_recurrent(vector<Tensor*> tin);
	void reset_loss();
	void print_loss(int b);
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


	void fit(vtensor tin, vtensor tout, int batch_size, int epochs);
	void fit_recurrent(vtensor tin, vtensor tout, int batch_size, int epochs);
	void train_batch(vtensor X, vtensor Y, vind sind, int eval = 0);
	void evaluate(vtensor tin, vtensor tout);
	void evaluate_recurrent(vtensor tin, vtensor tout);
	vtensor predict(vtensor tin);


};


void collectTensor(Layer *l,string tname="output",int p=0);
void distributeTensor(Layer *l,string tname="output", int p=0);

#endif  //EDDL_NET_H
