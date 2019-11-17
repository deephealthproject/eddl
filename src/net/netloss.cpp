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
#include <string>
#include <vector>

#include "netloss.h"
#include "../layers/core/layer_core.h"


NetLoss::NetLoss(Layer* (*f)(vector<Layer *>),vector<Layer *> in,string name)
{
 this->name=name;

 input=in;

 for(int i=0;i<in.size();i++)
  ginput.push_back(new LInput(new Tensor(in[i]->output->getShape()),"graph_input"+i,DEV_CPU));

 fout=(*f)(ginput);

 graph=new Net(ginput,{fout});

 Net *sn=in[0]->net;
 graph->build(sn->optimizer->clone(),{new LMin()},{new MSum()},sn->cs);


}

NetLoss::~NetLoss() {
  delete graph;
}

float NetLoss::compute()
{

   graph->reset();
   graph->reset_grads();
   graph->forward(input);
   graph->delta();
   graph->backward();

   value=fout->output->sum()/fout->output->shape[0];

   //printf("%s: %1.3f -- ",name.c_str(),value);
   //fflush(stdout);

   for(int i=0;i<ginput.size();i++) {
     collectTensor(ginput[i],"grad");
     Tensor::copy(ginput[i]->delta,input[i]->delta);
     distributeTensor(input[i],"grad");
   }

   return value;

}















/////
