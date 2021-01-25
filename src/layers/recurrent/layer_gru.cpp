/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/



#include <cstdio>
#include <cstdio>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/recurrent/layer_recurrent.h"


using namespace std;

int LGRU::total_layers = 0;

LGRU::LGRU(vector<Layer *> parent, int units, bool mask_zeros, bool bidirectional, string name, int dev, int mem): MLayer(name, dev, mem) {

    // DRAFT VERSION

    this->units = units;
    this->bidirectional = bidirectional;
    this->mask_zeros = mask_zeros;

    isrecurrent = true;

    if (parent[0]->output->ndim != 2) msg("LGRU only works over 2D tensors", "LGRU");

    if (name.empty()) this->name = "GRU" + to_string(++total_layers);

    input = parent[0]->output;
    state_hidden = output = new Tensor(vector<int>{input->shape[0], units}, dev);

    states.push_back(state_hidden);

    Wz_x = new Tensor(vector<int>{input->shape[1], units}, dev);
    Wr_x = new Tensor(vector<int>{input->shape[1], units}, dev);
    Wh_x = new Tensor(vector<int>{input->shape[1], units}, dev);
    params.push_back(Wz_x);
    params.push_back(Wr_x);
    params.push_back(Wh_x);

    gWz_x = new Tensor(vector<int>{input->shape[1], units}, dev);
    gWr_x = new Tensor(vector<int>{input->shape[1], units}, dev);
    gWh_x = new Tensor(vector<int>{input->shape[1], units}, dev);
    gradients.push_back(gWz_x);
    gradients.push_back(gWr_x);
    gradients.push_back(gWh_x);

    Wz_hidden = new Tensor(vector<int>{units, units}, dev);
    Wr_hidden = new Tensor(vector<int>{units, units}, dev);
    Wh_hidden = new Tensor(vector<int>{units, units}, dev);
    params.push_back(Wz_hidden);
    params.push_back(Wr_hidden);
    params.push_back(Wh_hidden);

    gWz_hidden = new Tensor(vector<int>{units, units}, dev);
    gWr_hidden = new Tensor(vector<int>{units, units}, dev);
    gWh_hidden = new Tensor(vector<int>{units, units}, dev);
    gradients.push_back(gWz_hidden);
    gradients.push_back(gWr_hidden);
    gradients.push_back(gWh_hidden);

    zn_bias = new Tensor(vector<int>{units}, dev);
    params.push_back(zn_bias);
    gzn_bias = new Tensor(vector<int>{units}, dev);
    gradients.push_back(gzn_bias);

    rn_bias = new Tensor(vector<int>{units}, dev);
    params.push_back(rn_bias);
    grn_bias = new Tensor(vector<int>{units}, dev);
    gradients.push_back(grn_bias);

    hn_bias = new Tensor(vector<int>{units}, dev);
    params.push_back(hn_bias);
    ghn_bias = new Tensor(vector<int>{units}, dev);
    gradients.push_back(ghn_bias);

    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }
}

LGRU::~LGRU() {
    //delete state_c;
}

// RESIZE , MEM_DELTA states
void LGRU::mem_delta() {
    // Reserve space for delta
    if (delta == nullptr) {
        // Delete deltas
        //for (int i = 0; i < delta_states.size(); ++i) { delete delta_states[i]; delta_states[i] = nullptr;}
        delta_states.clear();
        delta_hidden = delta = Tensor::zeros(this->output->shape, this->output->device);

        delta_states.push_back(delta_hidden);

        if (this->verbosity_level >= 2) {
            std::cout << "Booked delta for: " + this->name << std::endl;
        }
    }
}

void LGRU::free_delta() {
    if (delta != nullptr) {
        // The Tensor destructor takes into account the device details
        delete delta;
        delta = nullptr;  // Ensure nullptr

        if (this->verbosity_level >= 2) {
            std::cout << "Deleted delta for: " + this->name << std::endl;
        }
    }
}


void LGRU::resize(int batch) {
    if (output != nullptr) {
        output->resize(batch);
    }
}

// virtual
void LGRU::forward() {
    if (mask_zeros) {
        mask = new Tensor({input->shape[0], 1}, dev);
        reduced_abs_sum(input, mask);

        Tensor::logical_not(mask, mask);
        if (parent.size() > 1) {
            Tensor *A = replicate_tensor(mask, units);

            prev_hidden = parent[1]->states[0]->clone(); //prev state_h

            Tensor::el_mult(A, prev_hidden, prev_hidden, 0);
            delete A;
        }
    }

    /*
     * z gate: zn = sigmoid(x * Wz_x + hidden_-1 * Wz_hidden + zn_bias)
     */
    zn = new Tensor({input->shape[0], units}, dev);
    Tensor::mult2D(parent[0]->output, 0, Wz_x, 0, zn, 0); // x * Wz_x
    if (parent.size() > 1) {
        // hidden_-1 * Wz_hidden
        Tensor::mult2D(parent[1]->states[0], 0, Wz_hidden, 0, zn, 1);
    }
    Tensor::sum2D_rowwise(zn, zn_bias, zn);
    tensorNN::Sigmoid(zn, zn);

    /*
     * r gate: rn = sigmoid(x * Wr_x + hidden_-1 * Wr_hidden + rn_bias)
     */
    rn = new Tensor({input->shape[0], units}, dev);
    Tensor::mult2D(parent[0]->output, 0, Wr_x, 0, rn, 0); // x * Wr_x
    if (parent.size() > 1) {
        // hidden_-1 * Wr_hidden
        Tensor::mult2D(parent[1]->states[0], 0, Wr_hidden, 0, rn, 1);
    }
    Tensor::sum2D_rowwise(rn, rn_bias, rn);
    tensorNN::Sigmoid(rn, rn);

    /*
     * h gate: hn = tanh(x * Wh_x + (rn * hidden_-1) * Wh_hidden + hn_bias)
     */
    hn = new Tensor({input->shape[0], units}, dev);
    Tensor::mult2D(parent[0]->output, 0, Wh_x, 0, hn, 0); // x * Wh_x
    if (parent.size() > 1) {
        // (rn * hidden_-1) * Wh_hidden
        rn_hidden = new Tensor({input->shape[0], units}, dev);
        Tensor::el_mult(rn, parent[1]->states[0], rn_hidden, 0);
        Tensor::mult2D(rn_hidden, 0, Wh_hidden, 0, hn, 1);
    }// else (rn * hidden_-1) * Wh_hidden = 0
    Tensor::sum2D_rowwise(hn, hn_bias, hn);
    tensorNN::Tanh(hn, hn);

    /*
     * Compute h output of the cell: state_hidden = (1 - zn) * hidden_-1 + zn * hn 
     */
    hidden_not_zn = new Tensor({input->shape[0], units}, dev);
    if (parent.size() > 1) {
        // hidden_not_zn = (1 - zn) * hidden_-1
        not_zn = &(1 - *zn);
        Tensor::el_mult(not_zn, parent[1]->states[0], hidden_not_zn, 0);
    } else {
        hidden_not_zn->fill_(0.0);
    }
    zn_hn = new Tensor({input->shape[0], units}, dev);
    Tensor::el_mult(zn, hn, zn_hn, 0); // zn_hn = zn * hn
    Tensor::add(1.0, hidden_not_zn, 1.0, zn_hn, state_hidden, 0); // Final output value

    if (mask_zeros) {
        Tensor::logical_not(mask, mask);

        Tensor *A = replicate_tensor(mask, units);

        Tensor::el_mult(A, state_hidden, state_hidden, 0);

        delete A;

        if (parent.size() > 1) {
            Tensor::inc(prev_hidden, state_hidden); // output = prev output when in=0

            delete prev_hidden;
        }
    }

    if (!mode) { // eval mode
        delete zn;
        delete rn;
        delete hn;
        delete rn_hidden;
        delete zn_hn;
        delete hidden_not_zn;
        delete not_zn;
        if (mask_zeros) delete mask;
    }
}

void LGRU::backward() {
    if (mask_zeros) {
        if (parent.size() > 1) {
            Tensor::logical_not(mask, mask);

            Tensor *A = replicate_tensor(mask, units);

            prev_hidden = delta_hidden->clone();

            Tensor::el_mult(A, prev_hidden, prev_hidden, 0);

            delete A;
        }
    }

    Tensor *d_h = new Tensor(delta->getShape(), dev);
    Tensor *d_z = new Tensor(delta->getShape(), dev);
    Tensor *d_hidden = new Tensor(delta->getShape(), dev);
    Tensor *daux = new Tensor(delta->getShape(), dev);

    
    /*
     * h gate
     */
    daux->fill_(0.0);
    Tensor::el_mult(delta, zn, d_z, 0); // d_z: delta for hn, pending to apply derivative
    tensorNN::D_Tanh(d_z, hn, d_z);  // d_z: delta for hn
    if (trainable) {
        // Update gradients
        // parent[0]->output = $x_t$
        Tensor::mult2D(parent[0]->output, 1, d_z, 0, gWh_x, 1);
        if (parent.size() > 1) {
            // rn_hidden = $r_t * h_{t-1}$
            Tensor::mult2D(rn_hidden, 1, d_z, 0, gWh_hidden, 1);
        }
        Tensor::reduce_sum2D(d_z, ghn_bias, 0, 1);
    }
    // Propagate delta to parent
    Tensor::mult2D(d_z, 0, Wh_x, 1, parent[0]->delta, 1);
    if (parent.size() > 1) {
        Tensor::mult2D(d_z, 0, Wh_hidden, 1, daux, 1);
        // daux is the delta for $h_{t-1}$
        // rn = $r_t$
        Tensor::el_mult(daux, rn, parent[1]->delta_states[0], 1);
    }

    /*
     * r gate
     */
    // parent[1]->states[0]: $h_{t-1}$
    if (parent.size() > 1) {
        Tensor::el_mult(daux, parent[1]->states[0], daux, 0);
    } else {
        daux->fill_(0.0);
    }
    tensorNN::D_Sigmoid(daux, rn, daux); // daux is the delta for $r_t$
    if (trainable) {
        // Update gradients
        // parent[0]->output = $x_t$
        Tensor::mult2D(parent[0]->output, 1, daux, 0, gWr_x, 1);
        if (parent.size() > 1) 
            // parent[1]->states[0]: $h_{t-1}$
            Tensor::mult2D(parent[1]->states[0], 1, daux, 0, gWr_hidden, 1);
        Tensor::reduce_sum2D(daux, grn_bias, 0, 1);
    }
    // Propagate delta to parent
    Tensor::mult2D(daux, 0, Wr_x, 1, parent[0]->delta, 1);
    if (parent.size() > 1) 
        // parent[1]->delta_states[0]: delta for $h_{t-1}$
        Tensor::mult2D(daux, 0, Wr_hidden, 1, parent[1]->delta_states[0], 1);

    /*
     * z gate
     */
    Tensor::el_mult(delta, hn, d_h, 0);
    if (parent.size() > 1) {
        // not_zn: (1 - $z_t$)
        Tensor::el_mult(delta, not_zn, parent[1]->delta_states[0], 1); // Propagate delta to parent from the child
        Tensor::el_mult(delta, parent[1]->states[0], d_hidden, 0);
    } else {
        d_hidden->fill_(0.0);
    }

    Tensor::sub(d_h, d_hidden, daux);
    // d_h - d_hidden
    tensorNN::D_Sigmoid(daux, zn, daux); // daux is the delta for $z_t$
    if (trainable) {
        // Update gradients
        // parent[0]->output = $x_t$
        Tensor::mult2D(parent[0]->output, 1, daux, 0, gWz_x, 1);
        if (parent.size() > 1) 
            // parent[1]->states[0]: $h_{t-1}$
            Tensor::mult2D(parent[1]->states[0], 1, daux, 0, gWz_hidden, 1);
        Tensor::reduce_sum2D(daux, gzn_bias, 0, 1);
    }
    // Propagate delta to parent
    Tensor::mult2D(daux, 0, Wz_x, 1, parent[0]->delta, 1);
    if (parent.size() > 1) 
        // parent[1]->delta_states[0]: delta for $h_{t-1}$
        Tensor::mult2D(daux, 0, Wz_hidden, 1, parent[1]->delta_states[0], 1);

    if (mask_zeros) {
        if (parent.size() > 1) {
            Tensor::logical_not(mask, mask);

            Tensor *A = replicate_tensor(mask, units);

            Tensor::el_mult(A, parent[1]->delta_states[0], parent[1]->delta_states[0], 0);
            delete A;

            Tensor::inc(prev_hidden, parent[1]->delta_states[0]);

            delete prev_hidden;
        }
        delete mask;
    }

    delete d_h;
    delete d_z;
    delete d_hidden;
    delete daux;
    delete rn;
    delete zn;
    delete hn;
    delete hidden_not_zn;
    delete zn_hn;
    if (parent.size() > 1) {
        delete not_zn;
        delete rn_hidden;
    }

}


Layer *LGRU::share(int c, int bs, vector<Layer *> p) {
    LGRU *n = new LGRU(p, units, mask_zeros, bidirectional, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;
    n->isshared=true;

    //share params
    for (int i = 0; i < n->params.size(); i++) delete n->params[i];
    n->params.clear();

    n->Wz_x = Wz_x;
    n->Wr_x = Wr_x;
    n->Wh_x = Wh_x;
    n->params.push_back(Wz_x);
    n->params.push_back(Wr_x);
    n->params.push_back(Wh_x);
    n->zn_bias = zn_bias;
    n->rn_bias = rn_bias;
    n->hn_bias = hn_bias;
    n->params.push_back(zn_bias);
    n->params.push_back(rn_bias);
    n->params.push_back(hn_bias);

    if (n->parent.size() > 1) {
        n->Wz_hidden = Wz_hidden;
        n->Wr_hidden = Wr_hidden;
        n->Wh_hidden = Wh_hidden;
        n->params.push_back(Wz_hidden);
        n->params.push_back(Wr_hidden);
        n->params.push_back(Wh_hidden);
    }

    //share gradients
    for (int i = 0; i < n->gradients.size(); i++) delete n->gradients[i];
    n->gradients.clear();

    n->gWz_x = gWz_x;
    n->gWr_x = gWr_x;
    n->gWh_x = gWh_x;
    n->gradients.push_back(gWz_x);
    n->gradients.push_back(gWr_x);
    n->gradients.push_back(gWh_x);
    n->gzn_bias = gzn_bias;
    n->grn_bias = grn_bias;
    n->ghn_bias = ghn_bias;
    n->gradients.push_back(gzn_bias);
    n->gradients.push_back(grn_bias);
    n->gradients.push_back(ghn_bias);
    if (n->parent.size() > 1) {
        n->gWz_hidden = gWz_hidden;
        n->gWr_hidden = gWr_hidden;
        n->gWh_hidden = gWh_hidden;
        n->gradients.push_back(gWz_hidden);
        n->gradients.push_back(gWr_hidden);
        n->gradients.push_back(gWh_hidden);
    }

    n->reg=reg;
    n->init=init;

    return n;
}

Layer *LGRU::clone(int c, int bs, vector<Layer *> p, int todev) {
    LGRU *n = new LGRU(p, units, mask_zeros, bidirectional,  name, todev, this->mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LGRU::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}