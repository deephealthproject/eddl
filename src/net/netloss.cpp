/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "eddl/net/netloss.h"
#include "eddl/layers/core/layer_core.h"


NetLoss::NetLoss(const std::function<Layer*(vector<Layer*>)>& f, vector<Layer*> in, string name)
{
    this->name=name;

    input=in;

    for(int i=0;i<in.size();i++){
        ginput.push_back(new LInput(new Tensor(in[i]->output->getShape()),"graph_input" + to_string(i), DEV_CPU, 0));
    }

    fout=f(ginput);

    graph=new Net(ginput,{fout});

    Net *sn=in[0]->net;

    graph->build(sn->optimizer->clone(),{new LMin()},{new MSum()},sn->cs);


}

NetLoss::NetLoss(const std::function<Layer*(Layer*)>& f, Layer *in, string name)
{
    this->name=name;

    input.push_back(in);

    ginput.push_back(new LInput(new Tensor(in->output->getShape()),"graph_input", DEV_CPU, 0));

    fout=f(ginput[0]);

    graph=new Net(ginput,{fout});

    Net *sn=in->net;

    graph->build(sn->optimizer->clone(),{new LMin()},{new MSum()},sn->cs);


}


NetLoss::~NetLoss() {
    delete graph;
}

float NetLoss::compute(){
    int size=fout->output->size/fout->output->shape[0];

    graph->reset();
    graph->reset_grads();
    graph->forward(input);
    graph->delta();

    collectTensor(fout,"output");
    value=fout->output->sum()/size;

    return value;
}
