
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
  private:
    Tensor *T;

  public:
    Tensor *TC;
    Tensor *TG;

    TestTensor(vector<int>shape);
    void ToGPU();
    void check(string s);

};


TestTensor::TestTensor(vector<int>shape)
{

    TC=new Tensor(shape, DEV_CPU);
    TG=new Tensor(shape, DEV_GPU);
    T=new Tensor(shape, DEV_CPU);
}

void TestTensor::ToGPU(){
  Tensor::copy(TC,TG);
}


void TestTensor::check(string s) {
  Tensor::copy(TG,T);
  int val=Tensor::equal(T,TC);

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
  A->TC->rand_uniform(1.0);
  A->ToGPU();
  A->check("copy");

  ///////////// SET //////////////////////
  A->TC->set(1.0);
  A->TG->set(1.0);

  A->check("set");

  //////////// MULT2D ///////////////////
  A->TC->rand_uniform(1.0);
  B->TC->rand_uniform(1.0);

  A->ToGPU();
  B->ToGPU();

  Tensor::mult2D(A->TC,0,B->TC,0,C->TC,0);
  Tensor::mult2D(A->TG,0,B->TG,0,C->TG,0);

  C->check("mult2D");

  //////////// SUM /////////////////////
  A->TC->rand_uniform(1.0);
  D->TC->rand_uniform(1.0);

  A->ToGPU();
  D->ToGPU();

  Tensor::sum(1.0,A->TC,1.0,D->TC,E->TC,0);
  Tensor::sum(1.0,A->TG,1.0,D->TG,E->TG,0);

  E->check("sum");


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
