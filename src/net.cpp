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
#include "utils.h"
#include "random.h"

#ifdef cGPU
#include "hardware/gpu/gpu_tensor.h"
#endif

#define VERBOSE 0

using namespace std;
using namespace std::chrono;

ostream &operator<<(ostream &os, const vector<int> shape) {
    int i;
    os << "(";
    for (i = 0; i < shape.size() - 1; ++i) {
        os << shape[i];
        os << "x";
    }
    os << shape[i] << ")";

    return os;
}


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
int isIn(Layer *l, vlayer vl, int &ind) {
    for (int i = 0; i < vl.size(); i++)
        if (l == vl[i]) {
            ind = i;
            return 1;
        }

    return 0;
}

/////////////////////////////////////////
int isInorig(Layer *l, vlayer vl, int &ind) {
    for (int i = 0; i < vl.size(); i++)
        if (l == vl[i]->orig) {
            ind = i;
            return 1;
        }

    return 0;
}



////////////////////////////////////
///// NET CLASS
////////////////////////////////////

Net::Net(vlayer in, vlayer out) {
    // Set input/outlayer
    lin = in;
    lout = out;
    batch_size=1;
    // Default optimizer
    optimizer = nullptr;
    name="model";
    tr_batches=0;
    flog_tr=nullptr;
    flog_ts=nullptr;

    // Walk through the pointers of all layers, to get a plain
    // vector with all the layers
    for (int i = 0; i < lin.size(); i++) {
        walk(lin[i]);
    }
    for (int i = 0; i < lout.size(); i++) {
        walk_back(lout[i]);
    }

    for (int i = 0; i < lout.size(); i++) {
        total_loss.push_back(0.0);
        total_metric.push_back(0.0);
    }
    build_randn_table();
}

Net::~Net()
{
  for(int i=0;i<snets.size();i++)
    for(int j=0;j<snets[i]->layers.size();j++) {
      //cout<<"delete "<<nets[i]->layers[j]->name<<"\n";
      delete snets[i]->layers[j];
    }

}

/////////////////////////////////////////
int Net::inNet(Layer *l) {
    // Check if the layer l is in the network
    for (int i = 0; i < layers.size(); i++)
        if (l == layers[i]) return 1;
    return 0;
}


/////////////////////////////////////////
void Net::walk(Layer *l) {
    // If this layer is not in the network, add it, as well as all its children (recursively)
    if (!inNet(l)) {
      //cout<<l->name<<"\n";
      if (l->orig!=nullptr) l->net=l->orig->net;
      else l->net=this;

      layers.push_back(l);
      for (int i = 0; i < l->child.size(); i++)
          walk(l->child[i]);
    }
}
/////////////////////////////////////////
void Net::walk_back(Layer *l) {
    // If this layer is not in the network, add it, as well as all its children (recursively)

    if (!inNet(l)) {
      //cout<<l->name<<"  BACK\n";
      layers.push_back(l);
    }
    for (int i = 0; i < l->parent.size(); i++)
        walk_back(l->parent[i]);

}

/////////////////////////////////////////
Layer *Net::getLayer(string name) {
    for (int i = 0; i != layers.size(); i++)
        if (name == layers[i]->name) return layers[i];

    msg("layer %s not found", "Net.getLayer");
    return nullptr;
}

/////////////////////////////////////////
string Net::summary() {
    std::stringstream ss;

    for (int i = 0; i < vfts.size(); i++) {
          ss << vfts[i]->name.c_str() << " ";
    }


    ss << "\n";
    for (int i = 0; i < vfts.size(); i++) {
        ss << vfts[i]->name << ": ";

        vector<int> si = vfts[i]->input->getShape();
        si.erase(si.begin());
        vector<int> so = vfts[i]->output->getShape();
        so.erase(so.begin());
        ss << si << "-->" << so << "\n";

    }

    return ss.str();
}

void Net::plot(string fname,string mode) {
    ofstream out("tmp.dot");
    int ind;
    string type = fname.substr(fname.find('.') + 1);
    string cmd;


    out << "digraph Model {\n";
    out << "rankdir="<<mode<<";\n";

    // plot layers
    for (int i = 0; i != layers.size(); i++)
        if ((!isIn(layers[i], lin, ind)) && (!isIn(layers[i], lout, ind)))
            out << layers[i]->plot(0) << "\n";

    // Input Layers
    for (int i = 0; i != lin.size(); i++)
        out << lin[i]->plot(1) << "\n";

    // Output Layers
    for (int i = 0; i != lout.size(); i++)
        out << lout[i]->plot(1) << "\n";

    //plot links
    for (int i = 0; i != layers.size(); i++)
        for (int j = 0; j < layers[i]->child.size(); j++)
              out << layers[i]->name << "->" << layers[i]->child[j]->name << "\n";

    out << "}\n";

    out.close();

    cmd = "dot -T " + type + " ./tmp.dot >" + "./" + fname;

    system(cmd.c_str());

}

/////////////////////////////////////////
void Net::setlogfile(string fname)
{
  string str=fname+"_tr.log";
  string sts=fname+"_ts.log";

  flog_tr=fopen(str.c_str(),"wt");
  if (flog_tr==nullptr) msg("error creating tr log file","Net.setlogfile");

  flog_ts=fopen(sts.c_str(),"wt");
  if (flog_ts==nullptr) msg("error creating ts log file","Net.setlogfile");
}

/////////////////////////////////////////
void Net::initialize() {
    for (int i = 0; i != layers.size(); i++)
        layers[i]->initialize();
}

/////////////////////////////////////////
void Net::reset() {
    for (int i = 0; i != layers.size(); i++)
        layers[i]->reset();
}

void Net::save(FILE *fe)
{
  for (int i = 0; i != layers.size(); i++)
    layers[i]->save(fe);
}

void Net::load(FILE *fe)
{
  for (int i = 0; i != layers.size(); i++)
    layers[i]->load(fe);
}


/////////////////////////////////////////
void Net::fts() {
    int i, j, k, n;
    vector<int> visit;
    vector<int> gin;

    //fprintf(stdout,"FTS:");
    for (i = 0; i < layers.size(); i++) {
        visit.push_back(0);
        gin.push_back(layers[i]->lin);
    }

    for (i = 0; i < layers.size(); i++) {

        for (j = 0; j < layers.size(); j++)
            if ((gin[j] == 0) && (!visit[j])) break;

        if (j == layers.size())
            msg("error recurrent net", "Net.fts");

        /*
          if (layers[j]->lout)
          fprintf(stdout,"%s-->",layers[j]->name.c_str());
          else
          fprintf(stdout,"%s |",layers[j]->name.c_str());
        */
        visit[j] = 1;
        vfts.push_back(layers[j]);

        for (k = 0; k < layers[j]->lout; k++)
            for (n = 0; n < layers.size(); n++)
                if (layers[n] == layers[j]->child[k]) gin[n]--;

    }
    //fprintf(stdout,"\n");
    if (VERBOSE) {
      cout<<"Forward sort:";
      for (i = 0; i < vfts.size(); i++)
        cout<<vfts[i]->name<<"-->";
      cout<<"\n";
      //getchar();
    }
}


/////////////////////////////////////////
void Net::bts() {
    int i, j, k, n;
    vector<int> visit;
    vector<int> gout;

    //fprintf(stdout,"BTS:");
    for (i = 0; i < layers.size(); i++) {
        visit.push_back(0);
        gout.push_back(layers[i]->lout);
    }

    for (i = 0; i < layers.size(); i++) {

        for (j = 0; j < layers.size(); j++)
            if ((gout[j] == 0) && (!visit[j])) break;

        if (j == layers.size())
            msg("error recurrent net in", "Net.bts");

        visit[j] = 1;
        vbts.push_back(layers[j]);

        for (k = 0; k < layers[j]->lin; k++)
            for (n = 0; n < layers.size(); n++)
                if (layers[n] == layers[j]->parent[k]) gout[n]--;

    }
}


/////////////////////////////////////////
//// BUILD FUNCS
/////////////////////////////////////////

void Net::build(Optimizer *opt, vloss lo, vmetrics me, CompServ *cs){
    build(opt, lo, me);
    set_compserv(cs);

    if (cs->type == "local") {
      if (snets[0]->dev == DEV_CPU)
        cout << "Net running on CPU\n";
      else if (snets[0]->dev < DEV_FPGA)
        cout << "Net running on GPU " << snets[0]->dev - DEV_GPU << "\n";
      else
        cout << "Net running on FPGA " << snets[0]->dev - DEV_FPGA << "\n";
    }

}


void Net::build(Optimizer *opt, vloss lo, vmetrics me) {
    fprintf(stdout, "Build net %s\n",name.c_str());

    if (lo.size() != lout.size())
        msg("Loss list size does not match output list", "Net.build");

    if (me.size() != lout.size())
        msg("Metric list size does not match output list", "Net.build");

    // check devices
    dev = -1;
    int ind;
    for (int i = 0; i < layers.size(); i++){
        // do not consider input layers, since they are always on CPU
        if (!isIn(layers[i], lin, ind)) {
            if (dev == -1) dev = layers[i]->dev;
            else {
                if (layers[i]->dev != dev)
                  msg("Net with layers in different devices", "Net.build");
            }
        }
    }

    // set optimizer
    optimizer = opt;
    optimizer->setlayers(layers);
    // Initialize fitting errors vector
    for (int i = 0; i < lo.size(); i++) {
        fiterr.push_back(0.0);
        fiterr.push_back(0.0);
    }
    // set loss functions and create targets tensors

    this->losses = vloss(lo);
    for (int i = 0; i < lo.size(); i++) {
        if (lo[i]->name == "soft_cross_entropy") lout[i]->delta_bp = 1;
        lout[i]->target = new Tensor(lout[i]->output->getShape(), dev);
    }

    // set metrics
    this->metrics = vmetrics(me);

    // forward sort
    fts();
    // backward sort
    bts();
    // random params
    initialize();
}

void Net::set_compserv(CompServ *cs){
    int todev;
    this->cs=cs;

    if (cs->type == "local") {

        if (cs->local_gpus.size() > 0) todev = DEV_GPU;
        else if (cs->local_fpgas.size() > 0) todev = DEV_FPGA;
        else todev = DEV_CPU;

        if (todev == DEV_CPU) {
            if (dev == DEV_CPU) {

                int nthreads = cs->local_threads;
                if (nthreads <= 0)
                    msg("Threads must be > 0", "Net.build");

                Eigen::initParallel();
                Eigen::setNbThreads(nthreads);

                snets.push_back(this);
            } else {
                msg("Net and Layers device missmatch", "Net.build");
            }
        } else if (todev < DEV_FPGA) {
#ifndef cGPU
            msg("EDDLL not compiled for GPU", "Net.build");
#else
            // split on multiple GPUs
        int ngpus=gpu_devices();
        if (ngpus==0) {
          msg("GPU devices not found","Net.build");
        }
        if (cs->local_gpus.size()>ngpus)
        {
          msg("GPU list on ComputingService is larger than available devices","Net.build");
        }

        fprintf(stderr,"Selecting GPUs from CS_GPU\n");
        for(int i=0;i<cs->local_gpus.size();i++)
          if (cs->local_gpus[i]) {
            devsel.push_back(i);
            fprintf(stderr,"GPU(%d) ",i);
          }

        fprintf(stderr,"\n");
        if (!devsel.size())
          msg("No gpu selected","Net.build");

        cout<<"split into "<<devsel.size()<<" GPUs devices\n";
        split(devsel.size(),DEV_GPU);
#endif
        } else {
            // split on multiple FPGAs
        }
    } else {
        msg("Distributed version not yet implemented", "Net.build");
    }

    // create input and output tensors (X,Y)
    for (int i = 0; i < snets.size(); i++) {
      for (int j = 0; j < snets[i]->lin.size(); j++)
          Xs[i].push_back(new Tensor(snets[i]->lin[j]->input->shape));
      for (int j = 0; j < snets[i]->lout.size(); j++)
          Ys[i].push_back(new Tensor(snets[i]->lout[j]->output->shape));
    }
}

void Net::split(int c, int todev) {
    int i, j, k, l;

    vlayer nlayers;
    vlayer nin;
    vlayer nout;
    int ind;
    vlayer par;

    int bs=1;
    int m=0;

    for (i = 0; i < c; i++) {
        cout << "Split " << i << "\n";

        nlayers.clear();
        nin.clear();
        nout.clear();

        if (i == c - 1) bs += m;

        // set inputs
        for (j = 0; j < lin.size(); j++)  {
            nin.push_back(lin[j]->clone(c, bs, par, todev + devsel[i]));
            nlayers.push_back(nin[j]);
        }

        // special layers that are not input of net but has not parents
        for (j = 0; j < layers.size(); j++)
          if ((layers[j]->lin==0)&&(!isIn(layers[j],lin,ind)))
            nlayers.push_back(layers[j]->clone(c, bs, par, todev + devsel[i]));


        // rest of layers
        for (k = 0; k < layers.size(); k++) {
            for (j = 0; j < layers.size(); j++) {
                if (!isInorig(layers[j], nlayers, ind)) {
                    vlayer par;
                    for (l = 0; l < layers[j]->parent.size(); l++) {
                        if (!isInorig(layers[j]->parent[l], nlayers, ind)) break;
                        else par.push_back(nlayers[ind]);
                    }
                    if (l == layers[j]->parent.size()) {
                        nlayers.push_back(layers[j]->clone(i, bs, par, todev + devsel[i]));
                    }
                }

            }
          }

        // set outputs
        for (j = 0; j < lout.size(); j++)
            if (isInorig(lout[j], nlayers, ind))
                nout.push_back(nlayers[ind]);

        // create new net
        snets.push_back(new Net(nin, nout));

        // build new net
        char cname[100];
        sprintf(cname,"snet_%d",i);
        snets[i]->name=cname;
        snets[i]->build(optimizer->clone(), losses, metrics);

        //summary();
        snets[i]->plot("kk.pdf","LR");
    }
}


void Net::resize(int b)
{
  int i,j;

  if (batch_size==b) return;

  batch_size=b;

  int c=snets.size();
  int bs,m;

  if (batch_size<c) {
    printf("=====> Warning: batch_size (%d) lower than compserv resources (%d)\n",batch_size,c);
    bs=1;
    m=0;
    c=batch_size;
  }
  else {
    bs = batch_size / c;
    m = batch_size % c;
  }

  //cout<<"Resizing net to batch_size="<<b<<"\n";

  for (j = 0; j < layers.size(); j++) {
      layers[j]->resize(batch_size);
  }

  for(i=0; i<c; i++) {
    Xs[i].clear();
    Ys[i].clear();

    if (i==c-1) bs+=m;
    snets[i]->batch_size=bs;
    for (j = 0; j < snets[i]->layers.size(); j++) {
        snets[i]->layers[j]->resize(bs);
      }

    for (j = 0; j < snets[i]->lin.size(); j++)
        Xs[i].push_back(new Tensor(snets[i]->lin[j]->input->shape));

    for (j = 0; j < snets[i]->lout.size(); j++)
        Ys[i].push_back(new Tensor(snets[i]->lout[j]->output->shape));
  }


}



/////////////////////////////////////////////////////////////////
///// NET LEVEL FUNCS
/////////////////////////////////////////////////////////////////

void Net::forward() {

    for (int i = 0; i < vfts.size(); i++) {
        vfts[i]->forward();
        if (VERBOSE) {
          cout << vfts[i]->name << "\n";
          fprintf(stdout, "  %s In:%f\n", vfts[i]->name.c_str(), vfts[i]->input->sum());
          fprintf(stdout, "  %s Out:%f\n", vfts[i]->name.c_str(), vfts[i]->output->sum());
          //getchar();
        }
    }
}


void Net::backward() {
    for (int i = 0; i < vbts.size(); i++) {

        vbts[i]->backward();
        if (VERBOSE) cout<<"BACK: "<<vbts[i]->name<<"delta:"<<vbts[i]->delta->sum()<<"\n";
      }

}

void Net::delta() {
    for (int i = 0; i < lout.size(); i++)
        losses[i]->delta(lout[i]->target, lout[i]->output, lout[i]->delta);

}

void Net::calcloss() {

    int p = 0;
    for (int i = 0; i < lout.size(); i++, p += 2) {
        // loss value
        fiterr[p] = losses[i]->value(lout[i]->target, lout[i]->output);
        // metric value
        fiterr[p + 1] = metrics[i]->value(lout[i]->target, lout[i]->output);

    }
}


void Net::applygrads() {

    if (VERBOSE) {
        for (int i = 0; i < vbts.size(); i++) {
            cout <<vbts[i]->name << "\n";
            fprintf(stdout, "  In:%f\n", vbts[i]->input->sum());
            fprintf(stdout, "  Out:%f\n", vbts[i]->output->sum());
            fprintf(stdout, "  Delta:%f\n", vbts[i]->delta->sum());
            for (int j = 0; j < vbts[i]->gradients.size(); j++) {
                fprintf(stdout, "  %f\n", vbts[i]->gradients[j]->sum());
            }
        }
        getchar();
    }

    optimizer->applygrads(batch_size);
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

/////////////////////////////////////////////////////////////////
///// HIGH LEVEL FUNCS
/////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
//////// SIMPLE atomic ops

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

  if (batch_size<comp)
    comp=batch_size;

  for (int i = 0; i < comp; i++) {
      // Thread params
      td[i].net = snets[i];
      td[i].eval = 0;

      // Call thread
      rc = pthread_create(&thr[i], nullptr, forward_t, (void *) (&td[i]));
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

  if (batch_size<comp)
    comp=batch_size;

  for (int i = 0; i < comp; i++) {
      // Thread params
      td[i].net = snets[i];
      td[i].eval = 0;

      // Call thread
      rc = pthread_create(&thr[i], nullptr, backward_t, (void *) (&td[i]));
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

  if (snets[0]->dev != DEV_CPU)
    for (int i = 0; i < comp; i++) {
        for (int j = 0; j < 2 * lout.size(); j++) {
            fiterr[j] += snets[i]->fiterr[j];
        }
    }

  inferenced_samples+=batch_size;


}

void Net::reset_grads()
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

      // Call thread
      rc = pthread_create(&thr[i], nullptr, reset_t, (void *) (&td[i]));
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

void Net::compute_loss()
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

      // Call thread
      rc = pthread_create(&thr[i], nullptr, calcloss_t, (void *) (&td[i]));
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


void Net::update()
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

      // Call thread
      rc = pthread_create(&thr[i], nullptr, update_t, (void *) (&td[i]));
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
//////// COMPLEX
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

        // Thread params
        td[i].net = snets[i];
        td[i].eval = eval;

        // Call thread
        rc = pthread_create(&thr[i], nullptr, train_batch_t, (void *) (&td[i]));
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

    // Sum all errors
    if (snets[0]->dev != DEV_CPU)
      for (int i = 0; i < comp; i++) {
          for (int j = 0; j < 2 * lout.size(); j++) {
              fiterr[j] += snets[i]->fiterr[j];
          }
      }

    int p=0;
    for (int k = 0; k < lout.size(); k++, p += 2) {
          total_loss[k] += fiterr[p];  // loss
          total_metric[k] += fiterr[p + 1];  // metric
          fiterr[p] = fiterr[p + 1] = 0.0;
    }
    inferenced_samples+=batch_size;

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
