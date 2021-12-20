/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
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

LGRU::LGRU(vector<Layer *> parent, int units, bool mask_zeros, bool bidirectional, string name, int dev, int mem): MLayer(name, dev, mem)
{
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
    Wn_x = new Tensor(vector<int>{input->shape[1], units}, dev);
    params.push_back(Wz_x);
    params.push_back(Wr_x);
    params.push_back(Wn_x);

    gWz_x = new Tensor(vector<int>{input->shape[1], units}, dev);
    gWr_x = new Tensor(vector<int>{input->shape[1], units}, dev);
    gWn_x = new Tensor(vector<int>{input->shape[1], units}, dev);
    gradients.push_back(gWz_x);
    gradients.push_back(gWr_x);
    gradients.push_back(gWn_x);

    Uz_h = new Tensor(vector<int>{units, units}, dev);
    Ur_h = new Tensor(vector<int>{units, units}, dev);
    Un_h = new Tensor(vector<int>{units, units}, dev);
    params.push_back(Uz_h);
    params.push_back(Ur_h);
    params.push_back(Un_h);

    gUz_h = new Tensor(vector<int>{units, units}, dev);
    gUr_h = new Tensor(vector<int>{units, units}, dev);
    gUn_h = new Tensor(vector<int>{units, units}, dev);
    gradients.push_back(gUz_h);
    gradients.push_back(gUr_h);
    gradients.push_back(gUn_h);

    bias_z_t = new Tensor(vector<int>{units}, dev);
    params.push_back(bias_z_t);
    g_bias_z_t = new Tensor(vector<int>{units}, dev);
    gradients.push_back(g_bias_z_t);

    bias_r_t = new Tensor(vector<int>{units}, dev);
    params.push_back(bias_r_t);
    g_bias_r_t = new Tensor(vector<int>{units}, dev);
    gradients.push_back(g_bias_r_t);

    bias_n_t = new Tensor(vector<int>{units}, dev);
    params.push_back(bias_n_t);
    g_bias_n_t = new Tensor(vector<int>{units}, dev);
    gradients.push_back(g_bias_n_t);

    bias_n_t_hidden = new Tensor(vector<int>{units}, dev);
    params.push_back(bias_n_t_hidden);
    g_bias_n_t_hidden = new Tensor(vector<int>{units}, dev);
    gradients.push_back(g_bias_n_t_hidden);

    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

    distributed_training = false;
    acc_gUz_h = acc_gWz_x = nullptr;
    acc_gUr_h = acc_gWr_x = nullptr;
    acc_gUn_h = acc_gWn_x = nullptr;
    acc_g_bias_z_t = acc_g_bias_r_t = acc_g_bias_n_t = acc_g_bias_n_t_hidden = nullptr;
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
    } else if (this->delta->shape[0] != this->output->shape[0]) {

        for (int i = 0; i < this->delta_states.size(); ++i)
            this->delta_states[i]->resize(this->output->shape[0]);
        // we do the previous for loop just in case, but in the current
        // implementation only next commented line is required
        //this->delta->resize(this->output->shape[0]);
        

        if (this->verbosity_level >= 2) {
            std::cout << "Resized delta for: " + this->name << std::endl;
        }
    }
}

void LGRU::free_delta() {
    if (delta != nullptr) {
        // The Tensor destructor takes into account the device details
        delete delta;
        delta = nullptr;  // Ensure nullptr

        delta_states.clear();

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
    From https://github.com/onnx/onnx/blob/master/docs/Operators.md#GRU
    In this comment:
        x_t is the input to the current timestep
            x_t is parent[0]->output
        h_{t-1} is the output of the previous time step or hidden state of the GRU
            h_{t-1} is parent[1]->states[0]
        (Wbz + Ubz) is embeded in the attribute bias_z_t
        (Wbr + Ubr) is embeded in the attribute bias_r_t
        Wbn is bias_n_t
        Ubn is bias_n_t_hidden

    - z_t = sigmoid(x_t * Wz_x.T + h_{t-1} * (Uz_h.T) + Wbz + Ubz)
    - r_t = sigmoid(x_t * Wr_x.T + h_{t-1} * (Ur_h.T) + Wbr + Ubr)
    if linear_before_reset == 0
        - n_t = tanh(x_t * Wn_x.T + (r_t (.) h_{t-1}) * Un_h.T + Wbn + Ubn))
    else # when linear_before_reset != 0
        - n_t = tanh(x_t * Wn_x.T + Wbn + r_t (.) (h_{t-1} * Un_h.T + Ubn))
    - h_t = (1 - z_t) (.) n_t + z_t (.) h_{t-1}

    We only implement the version for linear_before_reset != 0
    */

    /*
     * z_t = sigmoid(x_t * Wz_x.T + h_{t-1} * (Uz_h.T) + Wbz + Ubz)
     * z_t = sigmoid(x_t * Wz_x.T + h_{t-1} * (Uz_h.T) + bias_z_t)
     */
    z_t = new Tensor({input->shape[0], units}, dev);
    Tensor::mult2D(parent[0]->output, 0, Wz_x, 0, z_t, 0); // x * Wz_x
    if (parent.size() > 1) {
        Tensor::mult2D(parent[1]->states[0], 0, Uz_h, 0, z_t, 1);
    }
    Tensor::sum2D_rowwise(z_t, bias_z_t, z_t);
    tensorNN::Sigmoid(z_t, z_t);

    /*
     * r_t = sigmoid(x_t * Wr_x.T + h_{t-1} * (Ur_h.T) + Wbr + Ubr)
     * r_t = sigmoid(x_t * Wr_x.T + h_{t-1} * (Ur_h.T) + bias_r_t)
     */
    r_t = new Tensor({input->shape[0], units}, dev);
    Tensor::mult2D(parent[0]->output, 0, Wr_x, 0, r_t, 0); // x * Wr_x
    if (parent.size() > 1) {
        Tensor::mult2D(parent[1]->states[0], 0, Ur_h, 0, r_t, 1);
    }
    Tensor::sum2D_rowwise(r_t, bias_r_t, r_t);
    tensorNN::Sigmoid(r_t, r_t);

    /*
     * n_t = tanh(x_t * Wn_x.T + Wbn + r_t (.) (h_{t-1} * Un_h.T + Ubn))
     * n_t = tanh(x_t * Wn_x.T + bias_n_t + r_t (.) (h_{t-1} * Un_h.T + bias_n_t_hidden))
     * n_t = tanh(x_t * Wn_x.T + bias_n_t + r_t (.) n_t_hidden
     *       because n_t_hidden is h_{t-1} * Un_h.T + bias_n_t_hidden
     */
    n_t = new Tensor({input->shape[0], units}, dev);
    n_t_hidden = new Tensor({input->shape[0], units}, dev);
    Tensor::mult2D(parent[0]->output, 0, Wn_x, 0, n_t, 0); // x * Wh_x
    if (parent.size() > 1) {
        Tensor::mult2D(parent[1]->states[0], 0, Un_h, 0, n_t_hidden, 0);
    } else {
        n_t_hidden->fill_(0.0f);
    }
    Tensor::sum2D_rowwise(n_t_hidden, bias_n_t_hidden, n_t_hidden);
    Tensor::el_mult(r_t, n_t_hidden, n_t, 1);
    Tensor::sum2D_rowwise(n_t, bias_n_t, n_t);
    tensorNN::Tanh(n_t, n_t);

    /*
     * Compute h output of the cell:
     * state_hidden = h_t = (1 - z_t) (.) n_t + z_t (.) h_{t-1}
     */
    one_minus_z_t = new Tensor({input->shape[0], units}, dev);
    one_minus_z_t->fill_(1.0f);
    Tensor::sub(one_minus_z_t, z_t, one_minus_z_t);
    Tensor::el_mult(one_minus_z_t, n_t, state_hidden, 0); // final output value
    if (parent.size() > 1) {
        Tensor::el_mult(z_t, parent[1]->states[0], state_hidden, 1); // final output value updated
    }

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
        delete z_t;
        delete r_t;
        delete n_t;
        delete n_t_hidden;
        delete one_minus_z_t;
        if (parent.size() > 1) {
        }
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

    Tensor *d1 = new Tensor(delta->getShape(), dev);
    Tensor *d2 = new Tensor(delta->getShape(), dev);
    Tensor *daux = new Tensor(delta->getShape(), dev);

    /*
     * n gate
     */
    Tensor::el_mult(delta, one_minus_z_t, d1, 0);
    daux->fill_(0.0);
    tensorNN::D_Tanh(d1, n_t, daux);  // daux is delta * (1 - z_t) * tanh'(n_t)
    if (trainable) {
        // Update gradients
        // parent[0]->output is x_t
        Tensor::mult2D(parent[0]->output, 1, daux, 0, gWn_x, 1);
        if (parent.size() > 1) {
            Tensor::el_mult(r_t, parent[1]->states[0], d2, 0);
            Tensor::mult2D(d2, 1, daux, 0, gUn_h, 1);
            ////
            Tensor::el_mult(daux, r_t, d2, 0); // d2 is delta * (1 - z_t) * tanh'(n_t) * r_t
            Tensor::reduce_sum2D(d2, g_bias_n_t_hidden, 0, 1);
        }
        Tensor::reduce_sum2D(daux, g_bias_n_t, 0, 1);
    }
    // Propagate delta to parent
    // here daux must be delta * (1 - z_t) * tanh'(n_t)
    Tensor::mult2D(daux, 0, Wn_x, 1, parent[0]->delta, 1); // delta * (1 - z_t) * tanh'(n_t) * Wn_x
    if (parent.size() > 1) {
        Tensor::el_mult(daux, r_t, d1, 0); // d1 is now delta * (1 - z_t) * tanh'(n_t) * r_t
        Tensor::mult2D(d1, 0, Un_h, 1, parent[1]->delta_states[0], 1);
    }

    /*
     * r gate
     *
     * daux comming from previous block for gate h already contains delta * (1 - z_t) * tanh'(n_t)
     */
    //fprintf(stderr, "control passed at %s(%d)\n", __FILE__, __LINE__);
    Tensor::el_mult(daux, n_t_hidden, d1, 0);
    d2->fill_(0.0);
    tensorNN::D_Sigmoid(d1, r_t, d2);
    // now d2 is delta * (1 - z_t) * tanh'(n_t) * (U_n * h_{t-1} + bias_n_t_hidden) * sigmoid'(r_t)
    //
    if (trainable) {
        Tensor::mult2D(parent[0]->output, 1, d2, 0, gWr_x, 1);
        if (parent.size() > 1) {
            Tensor::mult2D(parent[1]->states[0], 1, d2, 0, gUr_h, 1);
        }
        Tensor::reduce_sum2D(d2, g_bias_r_t, 0, 1);
    }

    // Propagate delta to parent
    Tensor::mult2D(d2, 0, Wr_x, 1, parent[0]->delta, 1);
    if (parent.size() > 1) {
        Tensor::mult2D(d2,  0, Ur_h, 1, parent[1]->delta_states[0], 1);
    }

    /*
     * z gate
     *
     */
    if (parent.size() > 1) {
        Tensor::copy(parent[1]->states[0], d1);
    } else {
        d1->fill_(0.0);
    }
    Tensor::sub(d1, n_t, d2);
    Tensor::el_mult(delta, d2, d1, 0);
    daux->fill_(0.0);
    tensorNN::D_Sigmoid(d1, z_t, daux); // now daux is delta * (h_{t-1} - n_t) * sigmoid'(z_t)
    if (trainable) {
        // Update gradients
        // parent[0]->output is $x_t$
        Tensor::mult2D(parent[0]->output, 1, daux, 0, gWz_x, 1);
        if (parent.size() > 1) {
            // parent[1]->states[0]: $h_{t-1}$
            Tensor::mult2D(parent[1]->states[0], 1, daux, 0, gUz_h, 1);
        }
        Tensor::reduce_sum2D(daux, g_bias_z_t, 0, 1);
    }
    // Propagate delta to parent
    // here daux is delta * (h_{t-1} - n_t) * sigmoid'(z_t)
    Tensor::mult2D(daux, 0, Wz_x, 1, parent[0]->delta, 1);
    if (parent.size() > 1) {
        Tensor::mult2D(daux, 0, Uz_h, 1, parent[1]->delta_states[0], 1);
        Tensor::el_mult(delta, z_t, parent[1]->delta_states[0], 1);
    }

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

    delete d1;
    delete d2;
    delete daux;
    delete r_t;
    delete z_t;
    delete n_t;
    delete n_t_hidden;
    delete one_minus_z_t;
    if (parent.size() > 1) {
    }
}

void LGRU::update_weights(vector<Tensor*> weights) {
    if (weights.size() == 10) {
        Tensor::copy(weights[0], Wz_x);
        Tensor::copy(weights[1], Wr_x);
        Tensor::copy(weights[2], Wn_x);
        Tensor::copy(weights[3], Uz_h);
        Tensor::copy(weights[4], Ur_h);
        Tensor::copy(weights[5], Un_h);
        Tensor::copy(weights[6], bias_z_t);
        Tensor::copy(weights[7], bias_r_t);
        Tensor::copy(weights[8], bias_n_t);
        Tensor::copy(weights[9], bias_n_t_hidden);
    } else {
        cerr << "[WARNING - LGRU::update_weights] "
             << "Unexpected number of weights tensors recieved "
             << "(weights.size()=" << weights.size() << ")" << endl;
    }
}

void LGRU::accumulate_accumulated_gradients(vector<Tensor*> grads) {
    if (grads.size() == 10) {
        Wz_x->add_(grads[0]);
        Wr_x->add_(grads[1]);
        Wn_x->add_(grads[2]);
        Uz_h->add_(grads[3]);
        Ur_h->add_(grads[4]);
        Un_h->add_(grads[5]);
        bias_z_t->add_(grads[6]);
        bias_r_t->add_(grads[7]);
        bias_n_t->add_(grads[8]);
        bias_n_t_hidden->add_(grads[9]);
    } else {
        cerr << "[WARNING - LGRU::accumulate_accumulated_gradients] "
             << "Unexpected number of gradient tensors recieved "
             << "(grads.size()=" << grads.size() << ")" << endl;
    }
}

void LGRU::reset_accumulated_gradients() {
    acc_gWz_x->fill_(0.0);
    acc_gWr_x->fill_(0.0);
    acc_gWn_x->fill_(0.0);
    acc_gUz_h->fill_(0.0);
    acc_gUr_h->fill_(0.0);
    acc_gUn_h->fill_(0.0);
    acc_g_bias_z_t->fill_(0.0);
    acc_g_bias_r_t->fill_(0.0);
    acc_g_bias_n_t->fill_(0.0);
    acc_g_bias_n_t_hidden->fill_(0.0);
}

void LGRU::apply_accumulated_gradients() {
    Wz_x->add_(acc_gWz_x);
    Wr_x->add_(acc_gWr_x);
    Wn_x->add_(acc_gWn_x);
    Uz_h->add_(acc_gUz_h);
    Ur_h->add_(acc_gUr_h);
    Un_h->add_(acc_gUn_h);
    bias_z_t->add_(acc_g_bias_z_t);
    bias_r_t->add_(acc_g_bias_r_t);
    bias_n_t->add_(acc_g_bias_n_t);
    bias_n_t_hidden->add_(acc_g_bias_n_t_hidden);
}

void LGRU::enable_distributed() {
    distributed_training = true;

    // Initialize the accumlated gradients tensors
    acc_gWz_x = new Tensor(vector<int>{input->shape[1], units}, output->device);
    acc_gWr_x = new Tensor(vector<int>{input->shape[1], units}, output->device);
    acc_gWn_x = new Tensor(vector<int>{input->shape[1], units}, output->device);
    acc_gUz_h = new Tensor(vector<int>{units, units}, output->device);
    acc_gUr_h = new Tensor(vector<int>{units, units}, output->device);
    acc_gUn_h = new Tensor(vector<int>{units, units}, output->device);
    acc_g_bias_z_t = new Tensor(vector<int>{units}, output->device);
    acc_g_bias_r_t = new Tensor(vector<int>{units}, output->device);
    acc_g_bias_n_t = new Tensor(vector<int>{units}, output->device);
    acc_g_bias_n_t_hidden = new Tensor(vector<int>{units}, output->device);

    // Set accumlated gradients to zero
    reset_accumulated_gradients();

    acc_gradients.push_back(acc_gWz_x);
    acc_gradients.push_back(acc_gWr_x);
    acc_gradients.push_back(acc_gWn_x);
    acc_gradients.push_back(acc_gUz_h);
    acc_gradients.push_back(acc_gUr_h);
    acc_gradients.push_back(acc_gUn_h);
    acc_gradients.push_back(acc_g_bias_z_t);
    acc_gradients.push_back(acc_g_bias_r_t);
    acc_gradients.push_back(acc_g_bias_n_t);
    acc_gradients.push_back(acc_g_bias_n_t_hidden);
}

Layer *LGRU::share(int c, int bs, vector<Layer *> p) {
    LGRU *n = new LGRU(p, units, mask_zeros, bidirectional, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;
    n->isshared=true;

    //share params
    for (int i = 0; i < n->params.size(); i++) delete n->params[i];
    n->params.clear();

    n->Wz_x = this->Wz_x;
    n->Wr_x = this->Wr_x;
    n->Wn_x = this->Wn_x;
    n->params.push_back(n->Wz_x);
    n->params.push_back(n->Wr_x);
    n->params.push_back(n->Wn_x);
    n->bias_z_t = this->bias_z_t;
    n->bias_r_t = this->bias_r_t;
    n->bias_n_t = this->bias_n_t;
    n->bias_n_t_hidden = this->bias_n_t_hidden;
    n->params.push_back(n->bias_z_t);
    n->params.push_back(n->bias_r_t);
    n->params.push_back(n->bias_n_t);
    n->params.push_back(n->bias_n_t_hidden);

    if (n->parent.size() > 1) {
        n->Uz_h = this->Uz_h;
        n->Ur_h = this->Ur_h;
        n->Un_h = this->Un_h;
        n->params.push_back(n->Uz_h);
        n->params.push_back(n->Ur_h);
        n->params.push_back(n->Un_h);
    }

    //share gradients
    for (int i = 0; i < n->gradients.size(); i++) delete n->gradients[i];
    n->gradients.clear();

    n->gWz_x = this->gWz_x;
    n->gWr_x = this->gWr_x;
    n->gWn_x = this->gWn_x;
    n->gradients.push_back(n->gWz_x);
    n->gradients.push_back(n->gWr_x);
    n->gradients.push_back(n->gWn_x);
    n->g_bias_z_t = this->g_bias_z_t;
    n->g_bias_r_t = this->g_bias_r_t;
    n->g_bias_n_t = this->g_bias_n_t;
    n->g_bias_n_t_hidden = this->g_bias_n_t_hidden;
    n->gradients.push_back(n->g_bias_z_t);
    n->gradients.push_back(n->g_bias_r_t);
    n->gradients.push_back(n->g_bias_n_t);
    n->gradients.push_back(n->g_bias_n_t_hidden);
    if (n->parent.size() > 1) {
        n->gUz_h = this->gUz_h;
        n->gUr_h = this->gUr_h;
        n->gUn_h = this->gUn_h;
        n->gradients.push_back(n->gUz_h);
        n->gradients.push_back(n->gUr_h);
        n->gradients.push_back(n->gUn_h);
    }

    if (distributed_training) {
        n->acc_gradients.clear();

        n->acc_gWz_x = acc_gWz_x;
        n->acc_gWr_x = acc_gWr_x;
        n->acc_gWn_x = acc_gWn_x;
        n->acc_gUz_h = acc_gUz_h;
        n->acc_gUr_h = acc_gUr_h;
        n->acc_gUn_h = acc_gUn_h;
        n->acc_g_bias_z_t = acc_g_bias_z_t;
        n->acc_g_bias_r_t = acc_g_bias_r_t;
        n->acc_g_bias_n_t = acc_g_bias_n_t;
        n->acc_g_bias_n_t_hidden = acc_g_bias_n_t_hidden;

        n->acc_gradients.push_back(acc_gWz_x);
        n->acc_gradients.push_back(acc_gWr_x);
        n->acc_gradients.push_back(acc_gWn_x);
        n->acc_gradients.push_back(acc_gUz_h);
        n->acc_gradients.push_back(acc_gUr_h);
        n->acc_gradients.push_back(acc_gUn_h);
        n->acc_gradients.push_back(acc_g_bias_z_t);
        n->acc_gradients.push_back(acc_g_bias_r_t);
        n->acc_gradients.push_back(acc_g_bias_n_t);
        n->acc_gradients.push_back(acc_g_bias_n_t_hidden);
    }

    n->do_deletes = false;
    if (n->reg != nullptr) delete n->reg;
    n->reg = reg;
    if (n->init != nullptr) delete n->init;
    n->init = init;

    return n;
}

Layer *LGRU::clone(int c, int bs, vector<Layer *> p, int todev) {
    LGRU *n = new LGRU(p, units, mask_zeros, bidirectional,  name, todev, this->mem_level);
    n->orig = this;

    if (distributed_training)
        n->enable_distributed();

    // TODO: Implement

    return n;
}


string LGRU::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
