// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019
//           Roberto Paredes Palacios, <rparedes@dsic.upv.es>
//           Jon Ander Gómez, <jon@dsic.upv.es>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "eddl.h"


using namespace std;

extern ostream &operator<<(ostream &os, const vector<int> shape);

EDDL eddl;

////////////////////////////////////////////////////////
///// EDDL is a wrapper class to ease and define the API
////////////////////////////////////////////////////////

tensor EDDL::T(const initializer_list<int> &shape) {
    vector<int> s(shape.begin(), shape.end());
    return T(s);
}

tensor EDDL::T(const vector<int> shape) {
    return new LTensor(shape, DEV_CPU);
}

tensor EDDL::T(string fname) {
    return new LTensor(fname);
}


void EDDL::div(tensor t, float v) {
    t->input->div(v);
}
//////////////////////////////////////////////////////

layer EDDL::Input(const initializer_list<int> &shape) {
    return new LInput(new Tensor(shape), "input" + to_string(1 + LInput::input_created), DEV_CPU);
}

layer EDDL::Input(tensor t) {
    return new LInput(t->input, "input" + to_string(1 + LInput::input_created), DEV_CPU);
}

//////////////////////////////////////////////////////
layer EDDL::Dense(layer parent, int ndim) {
    return new LDense(parent, ndim, "dense" + to_string(1 + LDense::dense_created), DEV_CPU);
}

layer EDDL::Dense(layer parent, int ndim, string name) {
    return new LDense(parent, ndim, name, DEV_CPU);
}


//////////////////////////////////////////////////////
layer EDDL::Conv(layer parent, const initializer_list<int> &ks) {
    return new LConv(parent, ks, {1, 1}, "same", "conv" + to_string(1 + LConv::conv_created), DEV_CPU);
}

layer EDDL::Conv(layer parent, const initializer_list<int> &ks, const initializer_list<int> &st) {
    return new LConv(parent, ks, st, "same", "conv" + to_string(1 + LConv::conv_created), DEV_CPU);
}

layer EDDL::Conv(layer parent, const initializer_list<int> &ks, const initializer_list<int> &st, string p) {
    return new LConv(parent, ks, st, p, "conv" + to_string(1 + LConv::conv_created), DEV_CPU);
}

layer EDDL::Conv(layer parent, const initializer_list<int> &ks, string p) {
    return new LConv(parent, ks, {1, 1}, p, "conv" + to_string(1 + LConv::conv_created), DEV_CPU);
}

//////////////////////////////////////////////////////
layer EDDL::MPool(layer parent, const initializer_list<int> &ks) {
    return new LMPool(parent, ks, ks, "none", "mpool" + to_string(1 + LMPool::pool_created), DEV_CPU);
}

layer EDDL::MPool(layer parent, const initializer_list<int> &ks, const initializer_list<int> &st) {
    return new LMPool(parent, ks, st, "none", "mpool" + to_string(1 + LMPool::pool_created), DEV_CPU);
}

layer EDDL::MPool(layer parent, const initializer_list<int> &ks, const initializer_list<int> &st, string p) {
    return new LMPool(parent, ks, st, p, "mpool" + to_string(1 + LMPool::pool_created), DEV_CPU);
}

layer EDDL::MPool(layer parent, const initializer_list<int> &ks, string p) {
    return new LMPool(parent, ks, ks, p, "mpool" + to_string(1 + LMPool::pool_created), DEV_CPU);
}

//////////////////////////////////////////////////////
layer EDDL::Activation(layer parent, string act) {
    return new LActivation(parent, act, "activation" + to_string(1 + LActivation::activation_created), DEV_CPU);
}

layer EDDL::Activation(layer parent, string act, string name) {
    return new LActivation(parent, act, name, DEV_CPU);
}


//////////////////////////////////////////////////////
layer EDDL::Reshape(layer parent, const initializer_list<int> &shape) {
    vector<int> s(shape.begin(), shape.end());
    return new LReshape(parent, s, "reshape" + to_string(1 + LReshape::reshape_created), DEV_CPU);
}

layer EDDL::Reshape(layer parent, const initializer_list<int> &shape, string name) {
    return new LReshape(parent, shape, name, DEV_CPU);
}

/////////////////////////////////////////////////////////
layer EDDL::Drop(layer parent, float df) {
    return new LDrop(parent, df, "drop" + to_string(1 + LDrop::drop_created), DEV_CPU);
}

layer EDDL::Drop(layer parent, float df, string name) {
    return new LDrop(parent, df, name, DEV_CPU);
}

/////////////////////////////////////////////////////////

layer EDDL::Add(const initializer_list<layer> &init) {
    return new LAdd(vlayer(init.begin(), init.end()), "add" + to_string(1 + LAdd::add_created), DEV_CPU);
}

layer EDDL::Add(const initializer_list<layer> &init, string name) {
    return new LAdd(vlayer(init.begin(), init.end()), name, DEV_CPU);
}

////////////////////////////////////////////////////////

layer EDDL::Cat(const initializer_list<layer> &init) {
    return new LCat(vlayer(init.begin(), init.end()), "cat" + to_string(1 + LCat::cat_created), DEV_CPU);
}

layer EDDL::Cat(const initializer_list<layer> &init, string name) {
    return new LCat(vlayer(init.begin(), init.end()), name, DEV_CPU);
}


////////////

optimizer EDDL::SGD(const initializer_list<float> &p) {
    return new sgd(p);
}

void EDDL::change(optimizer o, const initializer_list<float> &p) {
    o->change(p);
}

/////////////////////////////////////////////////////////
model EDDL::Model(vlayer in, vlayer out) {
    return new Net(in, out);
}

///////////
compserv EDDL::CS_CPU(int th) {
    return new CompServ(th, {}, {});
}

compserv EDDL::CS_GPU(const initializer_list<int> &g) {
    return new CompServ(0, g, {});
}

compserv EDDL::CS_FGPA(const initializer_list<int> &f) {
    return new CompServ(0, {}, f);
}


////////////

void EDDL::info(model m) {
    m->info();
}

void EDDL::plot(model m, string fname) {
    m->plot(fname);
}

void EDDL::build(model net, optimizer o, const initializer_list<string> &c, const initializer_list<string> &m) {
    net->build(o, c, m);
}

void EDDL::build(model net, optimizer o, const initializer_list<string> &c, const initializer_list<string> &m,
                 CompServ *cs) {
    net->build(o, c, m, cs);
}

void EDDL::fit(model net, const initializer_list<LTensor *> &in, const initializer_list<LTensor *> &out, int batch,
               int epochs) {
    vltensor ltin = vltensor(in.begin(), in.end());
    vltensor ltout = vltensor(out.begin(), out.end());

    vtensor tin;
    for (int i = 0; i < ltin.size(); i++)
        tin.push_back(ltin[i]->input);

    vtensor tout;
    for (int i = 0; i < ltout.size(); i++)
        tout.push_back(ltout[i]->input);


    net->fit(tin, tout, batch, epochs);
}

void EDDL::evaluate(model net, const initializer_list<LTensor *> &in, const initializer_list<LTensor *> &out) {
    vltensor ltin = vltensor(in.begin(), in.end());
    vltensor ltout = vltensor(out.begin(), out.end());

    vtensor tin;
    for (int i = 0; i < ltin.size(); i++)
        tin.push_back(ltin[i]->input);

    vtensor tout;
    for (int i = 0; i < ltout.size(); i++)
        tout.push_back(ltout[i]->input);


    net->evaluate(tin, tout);
}


////

bool exist(string name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    }
    return false;
}

void EDDL::download_mnist() {
    string cmd;
    string trX = "trX.bin";
    string trY = "trY.bin";
    string tsX = "tsX.bin";
    string tsY = "tsY.bin";

    if ((!exist(trX)) || (!exist(trY)) || (!exist(tsX)) || (!exist(tsY))) {
        cmd = "wget https://www.dropbox.com/s/khrb3th2z6owd9t/trX.bin";
        int status = system(cmd.c_str());
        if (status < 0) {
            msg("wget must be installed", "eddl.download_mnist");
            exit(1);
        }

        cmd = "wget https://www.dropbox.com/s/m82hmmrg46kcugp/trY.bin";
        status = system(cmd.c_str());
        if (status < 0) {
            msg("wget must be installed", "eddl.download_mnist");
            exit(1);
        }
        cmd = "wget https://www.dropbox.com/s/7psutd4m4wna2d5/tsX.bin";
        status = system(cmd.c_str());
        if (status < 0) {
            msg("wget must be installed", "eddl.download_mnist");
            exit(1);
        }
        cmd = "wget https://www.dropbox.com/s/q0tnbjvaenb4tjs/tsY.bin";
        status = system(cmd.c_str());
        if (status < 0) {
            msg("wget must be installed", "eddl.download_mnist");
            exit(1);
        }

    }
}












//////
