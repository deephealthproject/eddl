/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
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
#include "eddl/net/net.h"
#include "eddl/utils.h"
#include "eddl/random.h"

#include "eddl/layers/core/layer_core.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#endif

using namespace std;
using namespace std::chrono;


/////////////////////////////////////////
void Net::fts() {
    int i, j, k, n;
    vector<int> visit;
    vector<int> gin;

    for (i = 0; i < layers.size(); i++) {
        visit.push_back(0);

        n=0;
        for (j = 0; j < layers[i]->parent.size(); j++)
          if (isIn(layers[i]->parent[j],layers,k)) n++;

        gin.push_back(n);
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

}


/////////////////////////////////////////
void Net::bts() {
    int i, j, k, n;
    vector<int> visit;
    vector<int> gout;


    //fprintf(stdout,"BTS:");
    for (i = 0; i < layers.size(); i++) {
        visit.push_back(0);
        n=0;
        for (j = 0; j < layers[i]->child.size(); j++)
          if (isIn(layers[i]->child[j],layers,k)) n++;
        gout.push_back(n);
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
        for (unsigned int j = 0; j < Xs[i].size(); ++j) delete Xs[i][j];
        for (unsigned int j = 0; j < Ys[i].size(); ++j) delete Ys[i][j];
        Xs[i].clear();
        Ys[i].clear();
    }

    snets.clear();

    set_compserv(cs, true);
}

void Net::toGPU(vector<int> g,int lsb,int mem){
    CompServ *cs=new CompServ(0, g, {},lsb,mem);

    for (int i = 0; i < snets.size(); i++) {
        for (unsigned int j = 0; j < Xs[i].size(); ++j) delete Xs[i][j];
        for (unsigned int j = 0; j < Ys[i].size(); ++j) delete Ys[i][j];
        Xs[i].clear();
        Ys[i].clear();
    }

    snets.clear();

    set_compserv(cs, true);
}

void Net::build(Optimizer *opt, vloss lo, vmetrics me, CompServ *cs,
                bool initialize, bool do_optimizer_delete, bool do_compserv_delete){
	onnx_pretrained = !initialize; // For controlling when to copy the weights to the snet

  if (isbuild) return;

  cout<<"Building "<<name<<endl;

  for(int i=0;i<layers.size();i++) {
    if (layers[i]->net!=this) {
      layers[i]->net->build(opt->clone(),{},{},cs,true);
    }
  }

  make_graph(opt, lo, me, initialize);
  this->do_optimizer_delete = do_optimizer_delete;

  set_compserv(cs, do_compserv_delete);

  isbuild=true;
}


void Net::make_graph(Optimizer *opt, vloss lo, vmetrics me, bool initialize) {

    // check devices
    dev = -1;
    int ind;

    for(int i=0; i<layers.size(); i++){
        if (layers[i]->isrecurrent) isrecurrent=true;
        if (layers[i]->isdecoder) isdecoder=true;

        // Set device // TODO: Rewrite this
        if (dev == -1) {
            dev = layers[i]->dev;
        } else {
            if (layers[i]->dev != dev) {
                msg("Net with layers in different devices", "Net.build");
            }
        }

        // Set params
        layers[i]->verbosity_level = this->verbosity_level;
    }
    // set optimizer
    this->optimizer = opt;
    this->optimizer->setlayers(layers);

    // set loss functions and create targets tensors
    if ((isdecoder)||(isencoder)) {
      decsize=lout.size()/lo.size();
      for(int i=0;i<decsize;i++)
        for(int j=0;j<lo.size();j++)
           losses.push_back(lo[j]->clone());
    }
    else losses = vloss(lo);

    for (int i = 0; i < losses.size(); i++) {
        if (losses[i]->name == "softmax_cross_entropy") lout[i]->delta_bp = 1;
        lout[i]->target = new Tensor(lout[i]->output->getShape(), dev);
    }
    // set metrics
    if (isdecoder) {
      for(int i=0;i<decsize;i++)
        for(int j=0;j<me.size();j++)
            this->metrics.push_back(me[j]->clone());
    } else {
        for(int j=0;j<me.size();j++)
            this->metrics.push_back(me[j]);
    }

    // forward sort
    fts();

    // backward sort
    bts();
    // random params
    if(initialize) do_initialize();
}

void Net::set_compserv(CompServ *cs, bool do_compserv_delete){
    int todev;
    this->cs = cs;
    this->do_compserv_delete = do_compserv_delete;

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
            msg("EDDL not compiled for GPU", "Net.set_compserv");
#else
        // split on multiple GPUs
        int ngpus=gpu_devices();
        if (ngpus==0) {
          msg("GPU devices not found","Net.set_compserv");
        }
        if (cs->local_gpus.size()>ngpus)
        {
          msg("GPU list on ComputingService is larger than available devices","Net.set_compserv");
        }


        for(int i=0;i<cs->local_gpus.size();i++)
          if (cs->local_gpus[i]) {
            devsel.push_back(i);
          }

        if (!devsel.size())
          msg("No gpu selected","Net.set_compserv");

        if (!cs->isshared) {
          split(devsel.size(),DEV_GPU);
        }


#endif
        } else {
            // split on multiple FPGAs
#ifndef cFPGA
        msg("EDDLL not compiled for FPGA", "Net.build");
#else
        int nfpgas=1;  //fpga_devices();

        if (nfpgas==0) msg("FPGA devices not found","Net.build");
        if (cs->local_fpgas.size()>nfpgas) msg("FPGA list on ComputingService is larger than available devices","Net.build");

        fprintf(stderr,"Selecting FPGAs from CS_FPGA\n");

        for(int i=0;i<cs->local_fpgas.size();i++)
          if (cs->local_fpgas[i]) {
            devsel.push_back(i);
            fprintf(stderr,"FPGA(%d) ",i);
          }

        fprintf(stderr,"\n");

        if (!devsel.size()) msg("No fpga selected","Net.build");

        cout<<"split into "<<devsel.size()<<" FPGAs devices\n";
        if (!cs->isshared) {
            split(devsel.size(),DEV_FPGA);
        }
#endif
      }
    }
    else {
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

    //clone net into CompServices
    for (i = 0; i < c; i++) {

        nlayers.clear();
        nin.clear();
        nout.clear();
        if (i == c - 1) bs += m;

        //clone layers into CompServices
        for(int j=0;j<vfts.size();j++) {
          if (!isInorig(vfts[j], nlayers, ind)) {

              vlayer par;
              for (l = 0; l < vfts[j]->parent.size(); l++)
                  if (isInorig(vfts[j]->parent[l], nlayers, ind))
                    par.push_back(nlayers[ind]);

              Layer *n;
              if (vfts[j]->isshared) {
                Layer *on;
                if (vfts[j]->orig->clones.size()>i) {
                  on=vfts[j]->orig->clones[i];
                }
                else {
                  // should have been cloned previously in build
                  msg("error","build");
                }
                n=on->share(0,1,par);
                n->name="share_"+on->name;
                n->orig=vfts[j];
              }
              else {
                n=vfts[j]->clone(i, bs, par, todev + devsel[i]);
                n->isdecoder=vfts[j]->isdecoder;
                n->name="clone_"+to_string(i)+vfts[j]->name;
                vfts[j]->clones.push_back(n);
              }

              nlayers.push_back(n);
              if (isIn(vfts[j],lin,ind)) nin.push_back(n);
              if (isIn(vfts[j],lout,ind)) nout.push_back(n);
          }
        }

        // create twin net on CS device
        snets.push_back(new Net(nin, nout));
        // build new net
        char cname[100];
        sprintf(cname,"snet_%d",i);
        snets[i]->name=cname;
        snets[i]->make_graph(optimizer->clone(), this->losses, this->metrics);
        if(onnx_pretrained){ //We need to copy the imported weights to each snet
            //printf("Copying from CPU to GPU\n");
            for(int i = 0; i < snets.size(); i++)
                for(int j = 0; j < layers.size(); j++)
                    layers[j]->copy(snets[i]->layers[j]);
        }
        snets[i]->plot("smodel.pdf","LR");
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

  for (j = 0; j < layers.size(); j++)
      layers[j]->resize(batch_size);

  for(i=0; i<c; i++) {
    for (unsigned int j = 0; j < Xs[i].size(); ++j) delete Xs[i][j];
    for (unsigned int j = 0; j < Ys[i].size(); ++j) delete Ys[i][j];
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

void Net::setTrainable(string lname, bool val)
{
  for(int i=0;i<layers.size();i++) {
    if (layers[i]->name==lname) {
      Layer *l=layers[i];
      l->trainable=val;

      for(int j=0;j<snets.size();j++) {
        for(int k=0;k<snets[j]->layers.size();k++)
          if (snets[j]->layers[k]->orig==l) {
            cout<<"Setting device layer "<<snets[j]->layers[k]->name<<" trainable="<<val<<endl;
            snets[j]->layers[k]->trainable=val;
          }
      }//snets
    }//if
  }//layers
}


void Net::removeLayer(string lname)
{
  for(int i=0;i<layers.size();i++) {
    if (layers[i]->name==lname) {
      cout<<"removing "<<lname<<endl;
      Layer *l=layers[i];

      for(int j=0;j<l->parent.size();j++) {
        Layer *p=l->parent[j];
        for(int k=0;k<p->child.size();k++) {
          if (p->child[k]==l) {
            p->child.erase(p->child.begin() + k);
          }
        }//child
        // create new outputs from parents
        lout.push_back(p);
      }//parent

      // remove lname from out if is in
      //layers.erase(layers.begin() + i);
      for(int j=0;j<lout.size();j++) {
        cout<<lout[j]->name<<endl;
        if (lout[j]->name==lname) lout.erase(lout.begin()+j);
      }
      // remove lname from list of layers
      layers.erase(layers.begin() + i);
      delete l;
      return;
    }//if
  }// for layers
}


Layer * Net::getLayer(string lname)
{
  for(int i=0;i<layers.size();i++) {
    if (layers[i]->name==lname) return layers[i];
  }

  msg("Layer "+lname+" not found in model","getLayer");

  return nullptr;
}

void Net::enable_distributed(){
    for(Layer* l : layers)
        l->enable_distributed();

    for (int i = 0; i < snets.size(); i++)
        for(Layer* l : snets[i]->layers)
                   l->enable_distributed();
}
