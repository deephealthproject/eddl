
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <limits>

#include "apis/eddl.h"

using namespace eddl;
using namespace std;

int main(int argc, char **argv) {
    int dev = DEV_GPU;

    // Creation ops ***********************************
    // TEST: ones
    cout << "\n" << "ones: =============" << endl;
    auto t = Tensor::ones({10} , dev);
    t->info();
    t->print();

    // TEST: zeros
    cout << "\n" << "zeros: =============" << endl;
    t = Tensor::zeros({10} , dev);
    t->info();
    t->print();

    // TEST: full
    cout << "\n" << "full: =============" << endl;
    t = Tensor::full({10, 5}, 7.0);
    t->info();
    t->print();

    // TEST: arange
    cout << "\n" << "arange: =============" << endl;
    t = Tensor::arange(1.0, 2.5, 0.5);
    t->info();
    t->print();

    // TEST: range
    cout << "\n" << "range: =============" << endl;
    t = Tensor::range(1.0, 4.0, 0.5);
    t->info();
    t->print();

    // TEST: Linear space
    cout << "\n" << "linspace: =============" << endl;
    t = Tensor::linspace(0.1, 1.0, 5);
    t->info();
    t->print();

    // TEST: Linear space
    cout << "\n" << "logspace: =============" << endl;
    t = Tensor::logspace(0.1, 1.0, 5, 10.0);
    t->info();
    t->print();

    // TEST: Linear space
    cout << "\n" << "eye: =============" << endl;
    t = Tensor::eye(3);
    t->info();
    t->print();

    // TEST: randn
    cout << "\n" << "randn: =============" << endl;
    t = Tensor::randn({5, 4} , dev);
    t->info();
    t->print();

    // Math ops ***********************************
    // TEST: ones
    cout << "\n" << "abs: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->abs_();
    t->print();

    // TEST: acos
    cout << "\n" << "acos: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->acos_();
    t->print();

    // TEST: asin
    cout << "\n" << "asin: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->asin_();
    t->print();

    // TEST: atan
    cout << "\n" << "atan: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->atan_();
    t->print();

    // TEST: ceil
    cout << "\n" << "ceil: =============" << endl;
    t = Tensor::range(-2.0, 2.0, 0.25);
    t->print();
    t->ceil_();
    t->print();

    // TEST: clamp
    cout << "\n" << "clamp: =============" << endl;
    t = Tensor::range(-2.0, 2.0, 0.25);
    t->print();
    t->clamp_(-1.0f, 1.0f);
    t->print();

    // TEST: clampmax
    cout << "\n" << "clampmax: =============" << endl;
    t = Tensor::range(-2.0, 2.0, 0.25);
    t->print();
    t->clampmax_(1.0f);
    t->print();

    // TEST: clampmin
    cout << "\n" << "clampmin: =============" << endl;
    t = Tensor::range(-2.0, 2.0, 0.25);
    t->print();
    t->clampmin_(-1.0f);
    t->print();


    // TEST: cos_
    cout << "\n" << "cos_: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->cos_();
    t->print();

    // TEST: cosh
    cout << "\n" << "cosh: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->cosh_();
    t->print();

    // TEST: div_
    cout << "\n" << "div: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->div_(3.f);
    t->print();

    // TEST: exp
    cout << "\n" << "exp: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->exp_();
    t->print();

    // TEST: floor
    cout << "\n" << "floor: =============" << endl;
    t = Tensor::range(-2.0, 2.0, 0.25);
    t->print();
    t->floor_();
    t->print();

    // TEST: log
    cout << "\n" << "log: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->log_();
    t->print();

    // TEST: log2
    cout << "\n" << "log2: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->log2_();
    t->print();

    // TEST: log10
    cout << "\n" << "log10: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->log10_();
    t->print();

    // TEST: logn
    cout << "\n" << "logn: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->logn_(10.0f);
    t->print();

    // TEST: max
    cout << "\n" << "max: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    cout << "Max: " << t->max() << endl;

    // TEST: min
    cout << "\n" << "min: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    cout << "min: " << t->min() << endl;

    // TEST: mod
    cout << "\n" << "mod: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->mod_(2.0f);
    t->print();

    // TEST: mult
    cout << "\n" << "mult: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->mult_(5.0f);
    t->print();

    // TEST: normalize
    cout << "\n" << "normalize: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->normalize_(-1.0f, 1.0f);
    t->print();

    // TEST: neg
    cout << "\n" << "neg: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->neg_();
    t->print();

    // TEST: pow
    cout << "\n" << "pow: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->pow_(2.0f);
    t->print();

    // TEST: reciprocal
    cout << "\n" << "reciprocal: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->reciprocal_();
    t->print();

    // TEST: remainder
    cout << "\n" << "remainder: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->remainder_(2.0f);
    t->print();

    // TEST: round
    cout << "\n" << "round: =============" << endl;
    t = Tensor::range(-2.0, 2.0, 0.25);
    t->print();
    t->round_();
    t->print();

    // TEST: rsqrt
    cout << "\n" << "rsqrt: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->rsqrt_();
    t->print();

    // TEST: sigmoid
    cout << "\n" << "sigmoid: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->sigmoid_();
    t->print();

    // TEST: sign
    cout << "\n" << "sign: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->sign_();
    t->print();

    // TEST: sin
    cout << "\n" << "sin: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->sin_();
    t->print();

    // TEST: sinh
    cout << "\n" << "sinh: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->sinh_();
    t->print();

    // TEST: sqr
    cout << "\n" << "sqr: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->sqr_();
    t->print();

    // TEST: sqrt
    cout << "\n" << "sqrt: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->sqrt_();
    t->print();

    // TEST: sub
    cout << "\n" << "sub: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->sub_(3.0f);
    t->print();

    // TEST: sub
    cout << "\n" << "sum: =============" << endl;
    t = Tensor::range(0.0, 10.0, 1.0);
    t->print();
    cout << "sum:" << t->sum() << endl;
    t->print();

    // TEST: sum_abs
    cout << "\n" << "sum: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    cout << "sum_abs:" << t->sum_abs() << endl;
    t->print();


    // TEST: tan
    cout << "\n" << "tan: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->tan_();
    t->print();

    // TEST: tanh
    cout << "\n" << "tanh: =============" << endl;
    t = Tensor::range(-5.0, 5.0, 1.0);
    t->print();
    t->tanh_();
    t->print();

    // TEST: trunc
    cout << "\n" << "trunc: =============" << endl;
    t = Tensor::range(-2.0, 2.0, 0.25);
    t->print();
    t->trunc_();
    t->print();

}

