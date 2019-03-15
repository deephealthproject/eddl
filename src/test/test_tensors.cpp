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

int main(int argc, char **argv)
{


  Tensor *X=new Tensor("tsX",0);
  X->info();
  Tensor *Y=new Tensor("tsY",0);
  //Y->print();
  Y->info();

  X->save("tsX.bin");
  Y->save("tsY.bin");
  
  exit(1);


  Tensor *A=new Tensor({4,1200},DEV_GPU);
  A->set(1.0);
  A->print();
  Tensor *B=new Tensor({4,2},DEV_GPU);
  B->set(2.0);
  B->print();

  LInput *l1=new LInput(A,DEV_GPU);
  LInput *l2=new LInput(B,DEV_GPU);

  LCat *add=new LCat({l1,l2},DEV_GPU);


  add->forward();

  add->output->print();
  /////
  add->delta->set(5.0);
  add->delta->print();

  l1->delta->set(1.0);
  l2->delta->set(1.0);

  add->backward();

  l1->delta->print();
  l2->delta->print();

  exit(1);


  int r=1025;
  int c=2048;
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
  fprintf(stderr,"%f\n",C->total_sum()/(C->tam));
  */

}











///////////
