/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/layer.h"
#include "eddl/layers/operators/layer_operators.h"

using namespace std;


////////////////////////////////////
///// BASE LAYER CLASS
////////////////////////////////////

Layer::Layer(string name, int dev, int mem, const string &name_id) {
    mode = TRMODE;
    target = delta = input = output = nullptr;
    this->name = normalize_layer_name(name);  // We need to normalize due to some problem with graphviz
    this->name_id = name_id;
    this->dev = dev;
    this->mem_level = mem;
    lin = lout = 0;
    delta_bp = 0;
    detached=false;
    isrecurrent=false;
    isshared=false;
    isnorm=false;
    trainable=true;
    iscloned=false;
    isdecoder=false;
    distributed_training=false;

    this->do_deletes = true;

    this->orig = nullptr;
    this->sorig = nullptr;
    this->net = nullptr;

    this->reg = nullptr;
    // init = new IGlorotNormal(1234);
    this->init = new IGlorotUniform(1234);  // Has problems with the drive dataset

    this->reference_counter = 0; // accounts how many nets are referencing this layer
}

Layer::~Layer(){
    // Note: nullptr are not really needed. However, I like to have this pointers pointing to "something" just in case
    for (auto _ : this->states) if (_ != output) delete _;
    if (output != nullptr) { delete output; output = nullptr; }
    this->states.clear();
    if (target != nullptr) { delete target; target = nullptr; }

    for (auto _ : this->delta_states) if (_ != delta) delete _;
    this->delta_states.clear();
    if (delta != nullptr)  { delete delta; delta = nullptr; }


//    if (orig!=nullptr) delete this->orig;
//    if (net!=nullptr) delete this->net;
    if (this->do_deletes) {
        if (this->reg  != nullptr) { delete this->reg;  this->reg  = nullptr; }
        if (this->init != nullptr) { delete this->init; this->init = nullptr; }
    }

    //params if any
    if (!isshared){
        for(int i=0;i<params.size();i++){
            delete params[i]; params[i] = nullptr;
        }
    }

    //gradients if any
    if (!isshared){
        for(int i=0;i<gradients.size();i++){
            delete gradients[i]; gradients[i] = nullptr;
        }
    }

    //gradients if any
    if (!isshared){
        for(int i=0;i<acc_gradients.size();i++){
            delete acc_gradients[i]; acc_gradients[i] = nullptr;
        }
    }

}

void Layer::initialize() {
    for (int i = 0; i != params.size(); i++) {
        init->apply(params[i]);
    }
}

void Layer::clamp(float min, float max){
    for (int i = 0; i != params.size(); i++) {
        params[i]->clamp_(min,max);
    }
}

void Layer::set_detach(){
    detached=true;
}


void Layer::check_target() {
    if (target==nullptr) target=new Tensor(output->getShape(),dev);
    else if (target->size!=output->size) {
        delete target;
        target=new Tensor(output->getShape(),dev);
    }
}

void Layer::mem_delta_parent(){
    for(auto &p : this->parent){
        p->mem_delta();
    }
}

void Layer::mem_delta(){
    // Reserve space for the delta
    if(this->delta == nullptr) {
        this->delta = Tensor::zeros(this->output->shape, this->output->device);
    } else if (this->delta->shape[0] != this->output->shape[0]) {
        this->delta->resize(this->output->shape[0]);
    }
}

void Layer::free_delta(){
    if(this->delta != nullptr){
        // The Tensor destructor takes into account the device details
        delete this->delta;
        this->delta = nullptr;  // Ensure nullptr

        if(this->verbosity_level >= 2){
            std::cout << "Deleted delta for: " + this->name << std::endl;
        }
    }
}

void Layer::set_mem_level(int mem){
    mem_level=mem;
}

int Layer::decrease_and_get_reference_counter(){
    return --reference_counter;
}

void Layer::increase_reference_counter(){
    reference_counter++;
}

string Layer::get_name_id(){
    return this->name_id;
}

void Layer::resize(int batch){
    if (output!=nullptr) {
        output->resize(batch);
    }
}

void Layer::setTrainable(bool value){
    trainable=value;
}

int Layer::get_trainable_params_count()
{
        return params.size();
}

void Layer::detach(Layer *l){
    for(int i=0;i<child.size();i++){
        if(child[i]==l) {
            child.erase(child.begin() + i);
            lout--;
        }
    }
}

void Layer::reset() {
    if ((!mem_level) && (delta!=nullptr)) {
        delta->fill_(0.0);
        for(int i=0;i<states.size();i++)
            states[i]->fill_(0.0);
        for(int i=0;i<delta_states.size();i++)
            delta_states[i]->fill_(0.0);
    }
    detached=false;
}

void Layer::zeroGrads() {
    for(int i=0;i<gradients.size();i++){
        gradients[i]->fill_(0.0);
    }
}

void Layer::setmode(int m) {
    mode = m;
}

vector<int> Layer::getShape(){
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
        for (int i = 0; i < parent.size(); i++){
            cout << parent[i]->name << "\n";
        }
    } else { cout << "No parent layers\n"; }

    if (child.size()) {
        cout << "Child layers:\n";
        for (int i = 0; i != child.size(); i++){
            cout << child[i]->name << "\n";
        }
    } else { cout << "No child layers\n"; }

    cout << "Input tensor:\n";
    input->info();

    if (params.size()) {
        cout << "Params:\n";
        for (int i = 0; i < params.size(); i++){
            params[i]->info();
        }
    } else { cout << "No params\n"; }

    cout << "Output tensor:\n";
    output->info();
    cout << "===============\n\n";
}


void Layer::copy(Layer *l2){
    for(int i=0;i<params.size();i++){
        Tensor::copy(params[i],l2->params[i]);
    }
}



////////////////////////////////////
///// LINEAR LAYERS
////////////////////////////////////
LinLayer::LinLayer(string name, int dev, int mem, const string &name_id) : Layer(name, dev, mem, name_id) {}

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
MLayer::MLayer(string name, int dev, int mem) : Layer(name, dev, mem) {}

void MLayer::addchild(Layer *l) {
    child.push_back(l);
    lout++;
}

void MLayer::addparent(Layer *l) {
    parent.push_back(l);
    lin++;
}
