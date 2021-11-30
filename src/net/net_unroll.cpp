/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
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

#define VERBOSE 0

using namespace std;
using namespace std::chrono;


bool check_rnn_forward(Layer *l) {

  bool frnn=false;

  for(int i=0;i<l->child.size();i++) {
    if (l->child[i]->isrecurrent) {frnn=true;break;}
    else frnn=check_rnn_forward(l->child[i]);
    if (frnn) break;
  }

  return frnn;
}

Net* Net::unroll(int inl, int outl){
    msg("not implemented error", "Net::unroll");
}

// Unroll Recurrent net
Net* Net::unroll_enc(int inl, int outl) {
  int i, j, k, l;

  vlayer *nlayers;
  vlayer *nin;
  vlayer *nout;
  int ind;
  vlayer par;
  vector<bool> frnn;

  cout<<"Recurrent net input sequence length="<<inl<<endl;

  vlayer backup(layers);

  // set vfts sort
  layers.clear();
  for(int i=0;i<vfts.size();i++)
    layers.push_back(vfts[i]);

  // check if rnn is in forward path
  for(int i=0;i<layers.size();i++)
    if (layers[i]->isrecurrent) frnn.push_back(true);
    else frnn.push_back(check_rnn_forward(layers[i]));

  // set sort first frnn
  vlayer lfrnn;
  for(int i=0;i<layers.size();i++)
    if (frnn[i]) lfrnn.push_back(layers[i]);

  for(int i=0;i<layers.size();i++)
    if (!frnn[i]) lfrnn.push_back(layers[i]);

  layers.clear();
  for(int i=0;i<lfrnn.size();i++)
    layers.push_back(lfrnn[i]);


  // re-check frnn with the new sort
  frnn.clear();
  for(int i=0;i<layers.size();i++)
    if (layers[i]->isrecurrent) frnn.push_back(true);
    else frnn.push_back(check_rnn_forward(layers[i]));



  // unroll inputs
  nin=new vlayer[inl];
  nlayers=new vlayer[inl];
  nout=new vlayer[inl];

  for (i = 0; i < inl; i++) {
    //input layers
    for (j = 0; j < lin.size(); j++)  {
      Layer * l = lin[j]->share(i, batch_size, par);
      nin[i].push_back(l);
      nlayers[i].push_back(nin[i][j]);
    }

    // rest of layers
    for (j = 0; j < layers.size(); j++) {
      if ((i>=(inl-outl))||(frnn[j])) {
        if (!isInorig(layers[j], nlayers[i], ind)) {
          vlayer par;
          for (l = 0; l < layers[j]->parent.size(); l++) {
            if (!isInorig(layers[j]->parent[l], nlayers[i], ind)) break;
            else par.push_back(nlayers[i][ind]);
          }
          if (l == layers[j]->parent.size()) {

            if ((layers[j]->isrecurrent)&&(i>0)) {
              par.push_back(nlayers[i-1][j]);
              Layer * l = layers[j]->share(i, batch_size, par);
              nlayers[i].push_back(l);
            }
            else {
              Layer * l = layers[j]->share(i, batch_size, par);
              nlayers[i].push_back(l);
            }
          }
          else msg("Unexpected error","unroll");
        }
      }
    }

  // set output layers
  if (i>=(inl-outl)) {
    for (j = 0; j < lout.size(); j++)
      if (isInorig(lout[j], nlayers[i], ind))
        nout[i].push_back(nlayers[i][ind]);
  }

}

/////
vlayer ninl;
vlayer noutl;
for (i = 0; i < inl; i++)
  for (j = 0; j < nin[i].size(); j++)
    ninl.push_back(nin[i][j]);
for (i = 0; i < inl; i++)
  for (j = 0; j < nout[i].size(); j++)
    noutl.push_back(nout[i][j]);

  delete [] nin;
  delete [] nlayers;
  delete [] nout;

Net *rnet=new Net(ninl, noutl);

layers.clear();
for(auto l:backup) layers.push_back(l);


return rnet;

}


// Unroll Recurrent net
Net* Net::unroll_enc_dec(int inl, int outl) {
    int i, j, k, l;

    vlayer *nlayers;
    vlayer *nin;
    vlayer *nout;
    int ind;
    vlayer par;
    vector<bool> frnn;


    std:cerr << "Recurrent net encoder input sequence length=" << inl << ", decoder output sequence length=" << outl << std::endl;
    vlayer backup(layers);

    // set vfts sort
    layers.clear();
    for(int i=0;i<vfts.size();i++)
        layers.push_back(vfts[i]);

    /*for(int i=0;i<layers.size();i++)
      cout<<layers[i]->name<<endl;
      cout<<endl;
     */

    // check if rnn is in forward path
    for(i=0;i<layers.size();i++)
        if (layers[i]->isrecurrent) frnn.push_back(true);
        else frnn.push_back(check_rnn_forward(layers[i]));

    /*for(int i=0;i<layers.size();i++)
      if (frnn[i]) cout<<layers[i]->name<<"-->";
      else cout<<layers[i]->name<<"X-->";
      cout<<endl;
     */

    // check decoder branch if any
    for(i=0;i<layers.size();i++)
        if (layers[i]->isdecoder) {frnn[i]=false;break;}

    for(;i<layers.size();i++)
        frnn[i]=false;

    // set sort first frnn
    vlayer lfrnn;
    for(i=0;i<layers.size();i++)
        if (frnn[i]) lfrnn.push_back(layers[i]);

    for(i=0;i<layers.size();i++)
        if (!frnn[i]) lfrnn.push_back(layers[i]);


    layers.clear();
    for(i=0;i<lfrnn.size();i++)
        layers.push_back(lfrnn[i]);

    // re-check frnn with the new sort
    frnn.clear();
    for(i=0;i<layers.size();i++) frnn.push_back(false);


    int encsize=0;
    for(int i=0;i<layers.size();i++) {
        if (layers[i]->isdecoder) break;
        else {
            encsize++;
            if (layers[i]->isrecurrent) frnn[i]=true;
            else frnn[i]=check_rnn_forward(layers[i]);
        }
    }

    /*for(int i=0;i<layers.size();i++)
      if (frnn[i]) cout<<layers[i]->name<<"-->";
      else cout<<layers[i]->name<<"X-->";
      cout<<endl;
     */
    // unroll inputs
    nin=new vlayer[inl+outl];
    nlayers=new vlayer[inl+outl];
    nout=new vlayer[outl];

    int size=inl+outl;
    int top=0;

    bool connected=false;
    din.clear();
    for (i = 0; i < size; i++) {

        //encoder input layers
        if (i<inl) {
            for (j = 0; j < lin.size(); j++)  {
                nin[i].push_back(lin[j]->share(i, batch_size, par));
                nlayers[i].push_back(nin[i][j]);
            }
        }
        

        // rest of layers
        for (j = 0; j < layers.size(); j++) {
            // Encoder unroll
            if ((i<(inl+top))&&(frnn[j])) {
                if (!isInorig(layers[j], nlayers[i], ind)) {
                    vlayer par;
                    for (l = 0; l < layers[j]->parent.size(); l++) {
                        if (!isInorig(layers[j]->parent[l], nlayers[i], ind)) break;
                        else par.push_back(nlayers[i][ind]);
                    }
                    if (l == layers[j]->parent.size()) {
                        if ((layers[j]->isrecurrent)&&(i>0)) {
                            par.push_back(nlayers[i-1][j]);
                            nlayers[i].push_back(layers[j]->share(i, batch_size, par));
                        }
                        else {
                            nlayers[i].push_back(layers[j]->share(i, batch_size, par));
                        }
                    }
                    else msg("Unexpected error","unroll");
                }
            }
            else if ((i>=(inl-top))&&(!frnn[j])) {

                // End-Dec transition in case of decoder
                if ((isdecoder)&&(layers[j]->lin==0)) {
                    vlayer par;
                    Layer *n=layers[j]->share(i-inl, batch_size, par);
                    nin[i].push_back(n);
                    din.push_back(n); // decoder inputs
                    nlayers[i].push_back(n);
                }
                else {
                    vlayer par;
                    for (l = 0; l < layers[j]->parent.size(); l++) {
                        if (isInorig(layers[j]->parent[l], nlayers[i], ind)) 
                          par.push_back(nlayers[i][ind]);
                    }

                    if ((l == layers[j]->parent.size())||(layers[j]->isdecoder)) {
                        if ((layers[j]->isrecurrent)&&(i>0)) {
                            if (i==inl) {
                                if (!connected) {
                                    int c=nlayers[i-1].size();
                                    par.push_back(nlayers[i-1][c-1]);
                                    connected=true; // end-dec connected
                                }
                            }
                            else {
                                par.push_back(nlayers[i-1][nlayers[i].size()]);
                            }

                            nlayers[i].push_back(layers[j]->share(i, batch_size, par));

                        }
                        else {
                            int backi=i;
                            while (par.size()<layers[j]->parent.size()) {
                              for (l = 0; l < layers[j]->parent.size(); l++) {
                               if (!isInorig(layers[j]->parent[l], nlayers[i], ind)) 
                                  if (isInorig(layers[j]->parent[l], nlayers[backi], ind)) {
                                    par.push_back(nlayers[backi][ind]);
                                  }
                              }
                              backi--;
                              if (backi<0) 
                                msg("Unexpected error","unroll");
                             }
                                
                            nlayers[i].push_back(layers[j]->share(i, batch_size, par));
                        }
                    }
                    else msg("Unexpected error","unroll");
                }
            }
        }

        // set output layers
        if (i>=(inl-top)) {
            for (j = 0; j < lout.size(); j++)
                if (isInorig(lout[j], nlayers[i], ind))
                    nout[(i-inl)+top].push_back(nlayers[i][ind]);
        }


    }

    /////
    vlayer ninl;
    vlayer noutl;
    for (i = 0; i < inl+outl; i++)
        for (j = 0; j < nin[i].size(); j++) {
            ninl.push_back(nin[i][j]);
        }

    for (i = 0; i < outl; i++)
        for (j = 0; j < nout[i].size(); j++)
            noutl.push_back(nout[i][j]);

    if (!decoder_teacher_training) {
      for (i = 0; i < outl-1; i++) {
        noutl[i]->addchild(ninl[inl+i+1]);
        ninl[inl+i+1]->addparent(noutl[i]);
        }
    }

    Net *rnet=new Net(ninl, noutl);

    rnet->din=din;

    delete [] nin;
    delete [] nlayers;
    delete [] nout;

    layers.clear();
    for(auto l:backup) layers.push_back(l);


    return rnet;
}




// Unroll Recurrent net
Net* Net::unroll_dec(int inl, int outl) {
    int i, j, k, l;

    vlayer *nlayers;
    vlayer *nin;
    vlayer *nout;
    int ind;
    vlayer par;
    vector<bool> frnn;

    cout<<"Recurrent net output sequence length="<<outl<<endl;

    vlayer backup(layers);


    // set vfts sort
    layers.clear();
    for(int i=0;i<vfts.size();i++) {
        layers.push_back(vfts[i]);
        frnn.push_back(true);
    }

    // check if rnn is in forward path
    for(i=0;i<layers.size();i++)
        if (layers[i]->isdecoder) break;
        else frnn[i]=false;

    /*
       for(j=0; j<layers.size();j++) {
       if (frnn[j]) cout<<layers[j]->name<<"X"<<"-->";
       else cout<<layers[j]->name<<"-->";
       }
       cout<<"\n";

       getchar();
     */

    // unroll inputs
    nin=new vlayer[inl+outl];
    nlayers=new vlayer[outl];
    nout=new vlayer[outl];

    din.clear();
    for (i = 0; i < outl; i++) {
        if (i==0) {
            for (j = 0; j < lin.size(); j++)  {
                Layer * l = lin[j]->share(i, batch_size, par);
                nin[i].push_back(l);
                nlayers[i].push_back(nin[i][j]);
            }
        }

        // rest of layers
        Layer *last;
        for (j = 0; j < layers.size(); j++) {
            if ((i==0)||(frnn[j])) {
                if (!isInorig(layers[j], nlayers[i], ind)) {
                    vlayer par;
                    for (l = 0; l < layers[j]->parent.size(); l++) {
                        if (!isInorig(layers[j]->parent[l], nlayers[i], ind)) break;
                        else par.push_back(nlayers[i][ind]);
                    }
                    if (l == layers[j]->parent.size()) {
                        if ((layers[j]->isrecurrent)&&(i>0)) {
                            par.push_back(nlayers[i-1][j]);
                            Layer * l = layers[j]->share(i, batch_size, par);
                            nlayers[i].push_back(l);
                        }
                        else {
                            Layer *n;
                            n=layers[j]->share(i, batch_size, par);
                            nlayers[i].push_back(n);
                            if (layers[j]->lin==0) {
                                nin[i].push_back(n);
                                din.push_back(n);
                            }
                        }
                    }
                    else msg("Unexpected error","unroll");
                }
            }
            else if (!frnn[j]) {
                nlayers[i].push_back(nlayers[i-1][j]);
            }
        }

        // set output layers
        for (j = 0; j < lout.size(); j++)
            if (isInorig(lout[j], nlayers[i], ind))
                nout[i].push_back(nlayers[i][ind]);
    }

    /////
    vlayer ninl;
    vlayer noutl;
    for (i = 0; i < inl+outl; i++)
        for (j = 0; j < nin[i].size(); j++)
            ninl.push_back(nin[i][j]);

    for (i = 0; i < outl; i++)
        for (j = 0; j < nout[i].size(); j++)
            noutl.push_back(nout[i][j]);

    if (!decoder_teacher_training) {
      for (i = 0; i < outl-1; i++) {
        noutl[i]->addchild(ninl[inl+i+1]);
        ninl[inl+i+1]->addparent(noutl[i]);
        }
    }

    Net *rnet=new Net(ninl, noutl);

    rnet->din=din;

    delete [] nin;
    delete [] nlayers;
    delete [] nout;

    layers.clear();
    for(auto l:backup) layers.push_back(l);


    return rnet;
}


//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////

void Net::build_rnet(int inl,int outl) {
  int i, j, k, n;
  int todev;
  bool do_unroll=false;

  // Check if it is necessary to unroll again
  if (rnet==nullptr) do_unroll=true;
  else {
    // check squence lengths and current unrolled rnet
    if ((isencoder)&&(isdecoder)) {
      if ( ((inl+outl)!=rnet->lin.size()) || (outl!=rnet->lout.size()) )
        do_unroll=true;
    }
    else if ((isencoder)&&(inl!=rnet->lin.size())) do_unroll=true;
    else if (outl!=rnet->lout.size()) do_unroll=true;
  }

  if (!do_unroll) return;

  // TODO: problems deleting unrolled on GPU
  //if (rnet!=nullptr) delete rnet


  ////////////////////////////////////////
  // Create an unrolled version on CPU
  ////////////////////////////////////////
  if ((isencoder)&&(isdecoder)) rnet=unroll_enc_dec(inl,outl);
  else if (!isdecoder) rnet=unroll_enc(inl,outl);
  else rnet=unroll_dec(inl,outl);

  for(i=0;i<rnet->layers.size();i++) {
     rnet->layers[i]->isrecurrent=false;
     rnet->layers[i]->net=rnet;
     rnet->layers[i]->sorig=rnet->layers[i]->orig;
     rnet->layers[i]->orig=rnet->layers[i];
   }
   rnet->isrecurrent=false;
   rnet->isdecoder=isdecoder;
   rnet->isencoder=isencoder;


   vloss lr;
   for(i=0;i<losses.size();i++) lr.push_back(losses[i]->clone());

   vmetrics mr;
   for(i=0;i<this->metrics.size();i++) mr.push_back(this->metrics[i]->clone());


   rnet->build(optimizer->share(), lr, mr, cs->share(), false, true, true);

   rnet->plot("rmodel.pdf","LR");
   rnet->name="rnet";

   if (cs->local_gpus.size() > 0) todev = DEV_GPU;
   else if (cs->local_fpgas.size() > 0) todev = DEV_FPGA;
   else todev = DEV_CPU;

   ////////////////////////////////////////
   // Create an unrolled version on Device
   ////////////////////////////////////////
   if (todev!=DEV_CPU) {
     std::cerr << "Unroll on device" << std::endl;
     // unroll CS devices and link
     for(i=0;i<snets.size();i++) {
       if ((isencoder)&&(isdecoder))
         rnet->snets.push_back(snets[i]->unroll_enc_dec(inl,outl));
       else if (!isdecoder)
         rnet->snets.push_back(snets[i]->unroll_enc(inl,outl));
       else rnet->snets.push_back(snets[i]->unroll_dec(inl,outl));

       rnet->snets[i]->isencoder=rnet->isencoder;
       rnet->snets[i]->isdecoder=rnet->isdecoder;

       /// TODO:
       // check Xs Ys...
       // resize method reserve memory for Xs Ys... lucky guy

       for(j=0;j<rnet->snets[i]->layers.size();j++) {
             rnet->snets[i]->layers[j]->isrecurrent=false;
       }
       rnet->snets[i]->isrecurrent=false;

       rnet->snets[i]->make_graph(snets[i]->optimizer->share(),lr,mr,false);
       rnet->snets[i]->plot("rsnet.pdf","LR");
       for(j=0;j<rnet->snets[i]->layers.size();j++) {
             rnet->snets[i]->layers[j]->orig=rnet->layers[j];
             rnet->snets[i]->layers[j]->net=rnet;
       }
     }
   }

   rnet->flog_tr=flog_tr;
   rnet->flog_ts=flog_ts;

   rnet->reset_loss();
   rnet->reset();
   rnet->reset_grads();

   fflush(stdout);


}
