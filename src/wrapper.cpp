//
// Created by Salva Carrión on 2019-04-10.
//

#include <iostream>

#include "wrapper.h"
#include "eddl.h"
#include "layers/layer.h"
#include <thread>

// Create Tensors
Tensor *Tensor_init(const int *shape, int shape_size) {
    vector<int> v(shape, shape + shape_size);
    return new Tensor(v, DEV_CPU);
}

int Tensor_device(Tensor *t) {
    return t->device;
}

int Tensor_ndim(Tensor *t) {
    return t->ndim;
}

int Tensor_length(Tensor *t) { // Data length
    return t->tam;
}

int *Tensor_shape(Tensor *t) {
    return &t->sizes[0];
}

float *Tensor_getData(Tensor *t) {
    return t->ptr;
}

void Tensor_point2data(Tensor *t, float *ptr) {
    int tam = 1;
    for (int i = 0; i < t->ndim; ++i) tam *= t->sizes[i];
    t->tam = tam;
    t->ptr = ptr;
}

// Create Layers
tensor LTensor_init(const int *shape, int shape_size) {
    vector<int> s(shape, shape + shape_size);
    return new LTensor(s, DEV_CPU);
}

tensor LTensor_init_fromfile(const char *fname) {
    return new LTensor(fname);
}

void LTensor_div(tensor t, float v) {
    t->input->div(v);
}

layer Input_init(Tensor *in, const char *name) {
    return new LInput(in, name, DEV_CPU);
}

layer Dense_init(layer parent, int ndim, const char *name) {
    return new LDense(parent, ndim, name, DEV_CPU);
}

layer Conv_init(layer parent, const int *ks, int ks_size, const int *st, int st_size, const char *p, const char *name) {
    vector<int> vks(ks, ks + ks_size);
    vector<int> vst(st, st + st_size);
    return new LConv(parent, vks, vst, p, name, DEV_CPU);
}

layer MPool_init(layer parent, const int *ks, int ks_size, const int *st, int st_size, const char *p,
                 const char *name) {
    vector<int> vks(ks, ks + ks_size);
    vector<int> vst(st, st + st_size);
    return new LMPool(parent, vks, vst, p, name, DEV_CPU);
}

layer Activation_init(layer parent, const char *act, const char *name) {
    return new LActivation(parent, act, name, DEV_CPU);
}

layer Reshape_init(layer parent, const int *shape, int shape_size, const char *name) {
    vector<int> vshape(shape, shape + shape_size);
    return new LReshape(parent, vshape, name, DEV_CPU);
}

layer Drop_init(layer parent, float df, const char *name) {
    return new LDrop(parent, df, name, DEV_CPU);
}

layer Add_init(Layer **parent, int parent_size, const char *name) {
    vector<Layer *> vparent;
    vparent.reserve(parent_size);
    for (int i = 0; i < parent_size; ++i) {
        vparent.push_back(parent[i]);
    }
    return new LAdd(vparent, name, DEV_CPU);
};

layer Cat_init(Layer **init, int init_size, const char *name) {
    vector<Layer *> vinit;
    vinit.reserve(init_size);
    for (int i = 0; i < init_size; ++i) {
        vinit.push_back(init[i]);
    }
    return new LCat(vinit, name, DEV_CPU);
}

model Model_init(Layer *in, int in_size, Layer *out, int out_size) {
    vector<Layer *> vin;
    vin.reserve(in_size);
    for (int i = 0; i < in_size; ++i) {
        vin.push_back(in++);
    }

    vector<Layer *> vout;
    vout.reserve(out_size);
    for (int i = 0; i < out_size; ++i) {
        vout.push_back(out++);
    }
    return EDDL::Model(vin, vout);
}

void plot(model m, const char *fname) {
    EDDL::plot(m, fname);
}

void info(model m) {
    EDDL::info(m);
}

void build(model net, optimizer opt, const char **c, int size_c, const char **m, int size_m, compserv cs) {
    vector<string> co, me;

    // Add pointer values to vector of strings
    for (int i = 0; i < size_c; ++i) { co.emplace_back(*c); }
    for (int i = 0; i < size_m; ++i) { me.emplace_back(*m); }

    net->build(opt, co, me, cs);
}


void fit(model m, Tensor *in, Tensor *out, int batch, int epochs) {
    vector<Tensor *> tin = {in};
    vector<Tensor *> tout = {out};
    m->fit(tin, tout, batch, epochs);
}

void evaluate(model m, Tensor *in, Tensor *out) {
    vector<Tensor *> tin = {in};
    vector<Tensor *> tout = {out};
    m->evaluate(tin, tout);
}

const char *Layer_name(layer l) {
    char *name = new char[l->name.length() + 1];
    strcpy(name, l->name.c_str());
    return name;
}

Tensor *Layer_input(layer l) {
    return l->input;
}

Tensor *Layer_output(layer l) {
    return l->output;
}


// Optimizers
optimizer SGD_init(float lr, float mu) {
    return new sgd({lr, mu});
}

// Computing service
compserv CS_CPU_init(int th) {
    return new CompServ(th, {}, {});
}