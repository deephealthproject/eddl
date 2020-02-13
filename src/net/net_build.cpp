/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
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

#ifdef cGPU
#include "../hardware/gpu/gpu_tensor.h"
#endif

#define VERBOSE 0

using namespace std;
using namespace std::chrono;


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
void Net::toCPU(int t){
    CompServ *cs=new CompServ(t, {}, {},0);

    for (int i = 0; i < snets.size(); i++) {
      Xs[i].clear();
      Ys[i].clear();
    }

    snets.clear();

    set_compserv(cs);

    if (cs->type == "local") {
      if (VERBOSE)  {
        if (snets[0]->dev == DEV_CPU)
          cout << "Net running on CPU\n";
        else if (snets[0]->dev < DEV_FPGA)
          cout << "Net running on GPU " << snets[0]->dev - DEV_GPU << "\n";
        else
          cout << "Net running on FPGA " << snets[0]->dev - DEV_FPGA << "\n";
      }
    }
}
void Net::toGPU(vector<int> g,int lsb,int mem){
    CompServ *cs=new CompServ(0, g, {},lsb,mem);

    for (int i = 0; i < snets.size(); i++) {
      Xs[i].clear();
      Ys[i].clear();
    }

    snets.clear();

    set_compserv(cs);

    if (VERBOSE) {
    if (cs->type == "local") {
      if (snets[0]->dev == DEV_CPU)
        cout << "Net running on CPU\n";
      else if (snets[0]->dev < DEV_FPGA)
        cout << "Net running on GPU " << snets[0]->dev - DEV_GPU << "\n";
      else
        cout << "Net running on FPGA " << snets[0]->dev - DEV_FPGA << "\n";
    }
  }
}

void Net::build(Optimizer *opt, vloss lo, vmetrics me, CompServ *cs, bool initialize){
	build(opt, lo, me, initialize);
    set_compserv(cs);


  if (VERBOSE) {
    if (cs->type == "local") {
      if (snets[0]->dev == DEV_CPU)
        cout << "Net running on CPU\n";
      else if (snets[0]->dev < DEV_FPGA)
        cout << "Net running on GPU " << snets[0]->dev - DEV_GPU << "\n";
      else
        cout << "Net running on FPGA " << snets[0]->dev - DEV_FPGA << "\n";
    }
  }
}


void Net::build(Optimizer *opt, vloss lo, vmetrics me, bool initialize) {
    if (VERBOSE) cout<<"Build net "<<name<<"\n";

    if (lo.size() != lout.size())
        msg("Loss list size does not match output list", "Net.build");

    if (me.size() != lout.size())
        msg("Metric list size does not match output list", "Net.build");

    // check devices
    dev = -1;
    int ind;


    for(int i=0; i<layers.size(); i++){

        // Set device // TODO: Rewrite this
        if (dev == -1) {
            dev = layers[i]->dev;
        } else {
            if (layers[i]->dev != dev) {
                msg("Net with layers in different devices", "Net.build");
            }
        }

        // Set params
        layers[i]->set_trainable(true);
        layers[i]->verbosity_level = this->verbosity_level;
    }

    // set optimizer
    optimizer = opt;
    optimizer->setlayers(layers);

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
    if(initialize) do_initialize();
}

void Net::set_compserv(CompServ *cs){
    int todev;
    this->cs=cs;

    mem_level=cs->mem_level;
    for(int i=0;i<layers.size();i++)
        layers[i]->set_mem_level(mem_level);

    if (cs->type == "local") {

        if (cs->local_gpus.size() > 0) todev = DEV_GPU;
        else if (cs->local_fpgas.size() > 0) todev = DEV_FPGA;
        else todev = DEV_CPU;

        if (todev == DEV_CPU) {
            if (dev == DEV_CPU) {

                int nthreads = cs->local_threads;
                if (nthreads <= 0)
                    msg("Threads must be > 0", "Net.set_compserv");

                Eigen::initParallel();
                Eigen::setNbThreads(nthreads);

                snets.push_back(this);
            } else {
                msg("Net and Layers device missmatch", "Net.set_compserv");
            }
        } else if (todev < DEV_FPGA) {
#ifndef cGPU
            msg("EDDLL not compiled for GPU", "Net.set_compserv");
#else
            // split on multiple GPUs
        int ngpus=gpu_devices();
        if (ngpus==0) {
          msg("GPU devices not found","Net.set_compserv(");
        }
        if (cs->local_gpus.size()>ngpus)
        {
          msg("GPU list on ComputingService is larger than available devices","Net.set_compserv");
        }

        if (VERBOSE) cout<<"Selecting GPUs from CS_GPU\n";

        for(int i=0;i<cs->local_gpus.size();i++)
          if (cs->local_gpus[i]) {
            devsel.push_back(i);
            if (VERBOSE) cout<<"GPU ("<<i<<")\n";
          }


        if (!devsel.size())
          msg("No gpu selected","Net.set_compserv");

          if (VERBOSE) cout<<"split into "<<devsel.size()<<" GPUs devices\n";
        split(devsel.size(),DEV_GPU);
#endif
        } else {
            // split on multiple FPGAs
        }
    } else {
        msg("Distributed version not yet implemented", "Net.set_compserv");
    }

    // create input and output tensors (X,Y)
    for (int i = 0; i < snets.size(); i++) {
      for (int j = 0; j < snets[i]->lin.size(); j++)
          Xs[i].push_back(new Tensor(snets[i]->lin[j]->input->shape));
      for (int j = 0; j < snets[i]->lout.size(); j++)
          Ys[i].push_back(new Tensor(snets[i]->lout[j]->output->shape));
    }
}

// Split nets among CS
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
          if (VERBOSE) cout << "Split " << i << "\n";

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
        // for instance noise generators in GANs
        for (j = 0; j < layers.size(); j++)
          if ((layers[j]->lin==0)&&(!isIn(layers[j],lin,ind))) {
            nlayers.push_back(layers[j]->clone(c, bs, par, todev + devsel[i]));
          }

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

        // create twin net on CS device
        snets.push_back(new Net(nin, nout));

        // build new net
        char cname[100];
        sprintf(cname,"snet_%d",i);
        snets[i]->name=cname;
        snets[i]->build(optimizer->clone(), losses, metrics);

    }
}


void Net::resize(int b)
{
  int i,j;

  if (batch_size==b) return;

  batch_size=b;
  if (VERBOSE) cout<<"Resizing Net to batch_size="<<batch_size<<"\n";

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

  reset();

}

void Net::enable_distributed(){
	for(Layer* l : layers){
		l->enable_distributed();
	}
}
