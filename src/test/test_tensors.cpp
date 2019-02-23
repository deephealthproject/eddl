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

#include "../eddl.h"

int main(int argc, char **argv)
{

  int r=1024;
  int c=4;
  //
  Tensor *A=new Tensor({r,c},DEV_GPU);

  int r2,c2;

  r2=(A->tam/1024);


  if (r2==0) {
    r2=1;
    c2=A->tam;
  }
  else {
    if (A->tam%1024) r2++;
    c2=1024;
  }




  Tensor *B=new Tensor({r,c},DEV_GPU);
  Tensor *C=new Tensor({r,c},DEV_GPU);

  A->set(1.0);
  B->set(1.0);
  fprintf(stderr,"grid=%d block=%d\n",r2,c2);
  getchar();

  Tensor::sum(1.0,A,1.0,B,C,0);

  C->print();
  fprintf(stderr,"%f\n",C->total_sum()/(C->tam));


}











///////////
