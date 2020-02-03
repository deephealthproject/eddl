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

#include "layer.h"
#include "operators/layer_operators.h"

using namespace std;


////////////////////////////////////
///// BASE LAYER CLASS
////////////////////////////////////

Layer::Layer(string name, int dev) {
    mode = TRMODE;
    target = delta = input = output = nullptr;
    this->name = name;
    this->dev = dev;
    lin = lout = 0;
    delta_bp = 0;
    detached=false;

    orig=nullptr;
    net=nullptr;

    reg = nullptr;
    init=new IGlorotNormal(1234);
}

Layer::~Layer()
{
    if (output!=nullptr) delete output;
    if (delta!=nullptr) delete delta;
    if (target!=nullptr) delete target;

    //params if any
    for (int i=0;i<params.size();i++)
        delete params[i];

    //gradients if any
    for (int i=0;i<gradients.size();i++)
        delete gradients[i];

}

void Layer::initialize() {
    for (int i = 0; i != params.size(); i++) {
        init->apply(params[i]);
    }
}

void Layer::clamp(float min,float max)
{
  for (int i = 0; i != params.size(); i++) {
      params[i]->clamp_(min,max);
  }
}

void Layer::setdetach()
{
  detached=true;
}
void Layer::mem_delta()
{
  if (delta==nullptr) {
    delta=new Tensor(output->getShape(),output->device);
    delta->fill_(0.0);
  }
}
void Layer::free_delta()
{
  if (delta!=nullptr) {delete delta;delta=nullptr;}
}

  void Layer::setmem_level(int mem)
{
  mem_level=mem;
}

  void Layer::resize(int batch)
{
    //cout<<name<<" resizing\n";
    if (output!=nullptr) output->resize(batch);
    if (delta!=nullptr) delta->resize(batch);
    if (target!=nullptr) target->resize(batch);
}

void Layer::set_trainable(bool value)
{
  trainable=value;
}

void Layer::detach(Layer *l)
{
    for(int i=0;i<child.size();i++)
        if(child[i]==l) {
            child.erase(child.begin() + i);
            lout--;
        }
}

void Layer::reset() {
    if (mem_level<2) delta->fill_(0.0);
    detached=false;
}

void Layer::zeroGrads() {
  for(int i=0;i<gradients.size();i++)
    gradients[i]->fill_(0.0);
}

void Layer::setmode(int m) {

    mode = m;
}

vector<int> Layer::getShape()
{
    return output->getShape();
}

void Layer::save(std::ofstream &ofs, string format){
    for (int i = 0; i != params.size(); i++){
        params[i]->savefs(ofs, format);
    }
}

void Layer::load(std::ifstream &ifs, string format){
    for (int i = 0; i != params.size(); i++){
        Tensor* t=params[i]->loadfs(ifs, format);
        Tensor::copy(t,params[i]);
        delete t;
    }
}

void Layer::info() {
    cout << "\n===============\n";
    cout << "Layer " << name << "\n";
    if (parent.size()) {
        cout << "Parent layers:\n";
        for (int i = 0; i < parent.size(); i++)
            cout << parent[i]->name << "\n";
    } else cout << "No parent layers\n";

    if (child.size()) {
        cout << "Child layers:\n";
        for (int i = 0; i != child.size(); i++)
            cout << child[i]->name << "\n";
    } else cout << "No child layers\n";

    cout << "Input tensor:\n";
    input->info();

    if (params.size()) {
        cout << "Params:\n";
        for (int i = 0; i < params.size(); i++)
            params[i]->info();
    } else cout << "No params\n";

    cout << "Output tensor:\n";
    output->info();
    cout << "===============\n\n";
}

Tensor* Layer::getWeights(){
    return nullptr;
}

Tensor* Layer::setWeights(Tensor bias){
    return nullptr;
}

Tensor* Layer::getBias(){
    return nullptr;
}

Tensor* Layer::setBias(Tensor bias){
    return nullptr;
}


void Layer::copy(Layer *l2)
{
  for(int i=0;i<params.size();i++)
    Tensor::copy(params[i],l2->params[i]);
}

////////////////////////////////////
///// LINEAR LAYERS
////////////////////////////////////
LinLayer::LinLayer(string name, int dev) : Layer(name, dev) {}

void LinLayer::addchild(Layer *l) {
    child.push_back(l);
    lout++;
}

void LinLayer::addparent(Layer *l) {
    if (parent.size() != 0) msg("This layers only can have one parent layer", l->name.c_str());
    parent.push_back(l);
    lin++;
}


////////////////////////////////////
///// Multiple LAYERS
////////////////////////////////////
MLayer::MLayer(string name, int dev) : Layer(name, dev) {}

void MLayer::addchild(Layer *l) {
    child.push_back(l);
    lout++;
}

void MLayer::addparent(Layer *l) {
    parent.push_back(l);
    lin++;
}















/////
