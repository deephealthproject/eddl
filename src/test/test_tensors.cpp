// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019
// 	     Roberto Paredes Palacios, <rparedes@dsic.upv.es>
// 	     Jon Ander Gómez, <jon@dsic.upv.es>
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
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


#include "../eddl.h"

int main(int argc, char **argv) {

    int dev = DEV_CPU;

    Tensor *A = new Tensor({1, 1, 7, 7});
    A->set(1.0);
    A->ptr[1] = 10;

    A->print();


    ConvolDescriptor *cd = new ConvolDescriptor({1, 3, 3}, {1, 1}, "same");
    cd->build(A);

    cd->K->set(1.0);


    Tensor::Conv2D(cd);

    cd->O->print();
    shape s;
    s.push_back(cd->O->sizes[0] * cd->O->sizes[1] * cd->O->sizes[2]);
    s.push_back(cd->O->sizes[3]);
    Tensor *Cr = new Tensor(s, cd->O);
    Cr->print();


    PoolDescriptor *pd = new PoolDescriptor({3, 3}, {3, 3}, "none");
    pd->build(cd->O);

    pd->O->info();

    Tensor::MPool2D(pd);

    pd->O->print();
    shape s2;
    s2.push_back(pd->O->sizes[0] * pd->O->sizes[1] * pd->O->sizes[2]);
    s2.push_back(pd->O->sizes[3]);
    Tensor *Pr = new Tensor(s2, pd->O);
    Pr->print();

    cout << "ok\n";
    exit(1);


    //

    /*
    Tensor *A=new Tensor({r,c},DEV_GPU);
    A->set(1.0);
    A->print();
    getchar();


    Tensor *B=new Tensor({r,c},DEV_GPU);
    B->set(1.0);
    B->print();
    getchar();


    Tensor *C=new Tensor({r,c},DEV_GPU);
    Tensor::sum(1.0,A,1.0,B,C,0);

    C->print();
    fprintf(stderr,"%f\n",C->total_sum()/(C->size));
    */

}











///////////
