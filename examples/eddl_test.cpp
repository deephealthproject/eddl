
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
#include <cmath>

#include "apis/eddl.h"
#include "utils.h"

//////////////////////////////////////////////////////////
///////////// TestTensor class to ease testing ///////////
///////////// CPU, GPU ///////////////////////////////////
//////////////////////////////////////////////////////////
class TestTensor
{
  public:
    Tensor *T;
    Tensor *Tg;
    Tensor *Tc;

    TestTensor(vector<int>shape);
    void ToGPU();

};


TestTensor::TestTensor(vector<int>shape)
{

    T=new Tensor(shape, DEV_CPU);
    Tg=new Tensor(shape, DEV_GPU);
    Tc=new Tensor(shape, DEV_CPU);
}

void TestTensor::ToGPU(){
  Tensor::copy(T,Tg);
}


void check(TestTensor *A, string s) {
  Tensor::copy(A->Tg,A->Tc);
  int val=Tensor::equal(A->T,A->Tc);

  cout<<"====================\n";
  if (!val) {
    cout<<"Fail "<<s<<"\n";
    exit(1);
  }
  else {
    cout<<"OK "<<s<<"\n";
  }
  cout<<"====================\n";
}


//////////////////////////////////////////
//////////// TEST MAIN ///////////////////
//////////////////////////////////////////
int main(int argc, char **argv) {
  TestTensor *A=new TestTensor({10,10});
  TestTensor *B=new TestTensor({10,100});
  TestTensor *C=new TestTensor({10,100});

  TestTensor *D=new TestTensor({10,10});
  TestTensor *E=new TestTensor({10,10});

  ////////////// COPY ////////////////////

  A->T->rand_uniform(1.0);
  Tensor::copy(A->T,A->Tg);
  check(A,"copy");

  ///////////// SET //////////////////////

  A->T->set(1.0);
  A->Tg->set(1.0);

  check(A,"set");

  //////////// MULT2D ///////////////////
  A->T->rand_uniform(1.0);
  B->T->rand_uniform(1.0);

  A->ToGPU();
  B->ToGPU();

  Tensor::mult2D(A->T,0,B->T,0,C->T,0);
  Tensor::mult2D(A->Tg,0,B->Tg,0,C->Tg,0);

  check(C,"mult2D");

  //////////// SUM /////////////////////
  D->T->rand_uniform(1.0);
  D->ToGPU();

  Tensor::sum(1.0,A->T,1.0,D->T,E->T,0);
  Tensor::sum(1.0,A->Tg,1.0,D->Tg,E->Tg,0);

  check(E,"sum");



//
//    int dev = DEV_CPU;
//    Tensor *A=new Tensor({4,2,3,7}, dev);
//    Tensor *B=new Tensor({4,3}, dev);
//    Tensor *C=new Tensor({4,3}, dev);
//
//    vector<int> axis;
//    axis.push_back(1);
//    axis.push_back(3);
//
//    A->info();
//    A->set(1.0);
//    A->rand_uniform(1.0);
//    A->print();
//
//
//    cout<<"Mean\n";
//    Tensor::reduce(A,B,axis,"mean", false,NULL,0);
//
//    B->info();
//    B->print();
//
// /////
//    cout<<"Max\n";
//    Tensor::reduce(A,B,axis,"max",false,C,0);
//
//    B->info();
//    B->print();
//    C->info();
//    C->print();
//    cout<<"Delta max\n";
//    Tensor::delta_reduce(B,A,axis,"max",false,C,0);
//
//    A->print();
//
///////
//    cout<<"Sum\n";
//    Tensor::reduce(A,B,axis,"sum",false,NULL,0);
//    B->info();
//    B->print();
//
//    cout<<"==================\n";
//    cout<<"keepdims true\n";
//    cout<<"==================\n";
//    A=new Tensor({4,2,3});
//    B=new Tensor({4,2,3});
//    C=new Tensor({4,2,3});
//
//    vector<int> axis2;
//    axis2.push_back(1);
//
//    A->info();
//    A->set(1.0);
//    A->print();
//
//    cout<<"Mean\n";
//    Tensor::reduce(A,B,axis2,"mean",true,NULL,0);
//
//    B->info();
//    B->print();
//
//    /////
//    cout<<"Max\n";
//    Tensor::reduce(A,B,axis2,"max",true,C,0);
//
//    B->info();
//    B->print();
//    C->info();
//    C->print();
//
//    cout<<"Delta max\n";
//    Tensor::delta_reduce(B,A,axis2,"max",true,C,0);
//    A->print();
//
//    /////
//    cout<<"Sum\n";
//    Tensor::reduce(A,B,axis2,"sum",true,NULL,0);
//    B->info();
//    B->print();
//
//    cout<<"==================\n";
//    cout<<"EDDL Layers\n";
//    cout<<"==================\n";
//
//    tensor t =T({1,10,10,4});
//    t->data->set(1.0);
//    t->data->ptr[0]=10;
//
//    cout<<"\nMean\n";
//    layer m=ReduceMean(t,{1,2});
//    m->forward();
//    m->output->info();
//    m->output->print();
//
//    cout<<"\nVar\n";
//    //t->data->print();
//    layer v=ReduceVar(t,{1,3});
//
//    v->forward();
//    v->output->info();
//    v->output->print();



}


///////////
