//
// Created by Salva Carrión on 2019-04-10.
//

#include <iostream>

#include "wrapper.h"
#include "eddl.h"

//// For debugging
//std::cout << "T_init: " << "["; for ( int i = 0; i < init_size; i++) cout << *(init + i) << ", "; std::cout << "]" << std::endl;
//std::cout << "init_size: " << init_size << std::endl;
//std::cout << "dev: " << dev << std::endl;

tensor T_init(const int* init, int init_size, int dev){
    vector<int> v(init, init + init_size);
    return EDDL::T(v, dev);
}

layer Input_init(tensor t, int dev){
    return EDDL::Input(t, dev);
}

layer Dense_init(layer parent, int dim, const char* name, int dev){
    return EDDL::Dense(parent, dim, dev);
}

layer Conv_init(layer parent, const int* ks, int ks_size, const int* st, int st_size, const char* p, int dev){
    vector<int> vks(ks, ks + ks_size);
    vector<int> vst(st, st + st_size);
    return EDDL::Conv(parent, vks, vst, p, dev);
}

layer Activation_init(layer parent, const char* act, const char* name, int dev){
    return EDDL::Activation(parent, act, dev);
}

layer Reshape_init(layer parent, const int* init, int init_size, const char* name, int dev){
    vector<int> vinit(init, init + init_size);
    return EDDL::Reshape(parent, vinit, name, dev);
}

layer Drop_init(layer parent, float df, const char* name, int dev){
    return EDDL::Drop(parent, df, dev);
}

layer Add_init(Layer** parent, int parent_size, const char* name, int dev){
    vector<Layer*> vparent;
    vparent.reserve(parent_size);
    for( int i = 0; i < parent_size; ++i ){
        vparent.push_back(parent[i]);
    }
    return new LAdd(vparent, dev);
 };

layer Cat_init(Layer** init, int init_size, const char* name, int dev){
   vector<Layer*> vinit;
    vinit.reserve(init_size);
    for( int i = 0; i < init_size; ++i ){
        vinit.push_back(init[i]);
    }
    return new LCat(vinit, dev);
}

model Model_init(Layer* in, int in_size, Layer* out, int out_size){
    vector<Layer*> vin;
    vin.reserve(in_size);
    for( int i = 0; i < in_size; ++i ) {
        vin.push_back(in++);
    }

    vector<Layer*> vout;
    vout.reserve(out_size);
    for( int i = 0; i < out_size; ++i ) {
        vout.push_back(out++);
    }

    return EDDL::Model(vin, vout);
}

void plot(model m, const char* fname){
    EDDL::plot(m, fname);
}

void info(model m){
    EDDL::info(m);
}

void build(model net, optim *opt, const char** c, int size_c, const char** m, int size_m, int todev){
    vector<string> co, me;

    for(int i = 0; i < size_c; ++i){co.emplace_back(*c);}
    for(int i = 0; i < size_m; ++i){me.emplace_back(*m);}

    net->build(opt, co, me, todev);
}


void fit(model m, tensor in, tensor out, int batch, int epochs){
    vector<Tensor*> tin = {in->input};
    vector<Tensor*> tout = {out->input};

    m->fit(tin, tout, batch, epochs);
}

void evaluate(model m, tensor in, tensor out){
    vector<Tensor*> tin = {in->input};
    vector<Tensor*> tout = {out->input};

    m->evaluate(tin,tout);
}

const char* Layer_name(layer l){
    char* name = new char[l->name.length() + 1];
    strcpy(name, l->name.c_str());
    std::cout << "Cat layer address-g: " << &l << std::endl;
    return name;
}

// Optimizers
sgd* SGD_init(float lr, float mu){
    return SGD(lr, mu);
}
