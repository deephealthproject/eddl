
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
    TestTensor(Tensor *t_tc, Tensor *t_tg);
    void ToGPU();
    void check(string s);

};


TestTensor::TestTensor(vector<int>shape)
{

    TC=new Tensor(shape, DEV_CPU);
    TG=new Tensor(shape, DEV_GPU);
    T=new Tensor(shape, DEV_CPU);
}

TestTensor::TestTensor(Tensor *t_tc, Tensor *t_tg)
{
    TC=t_tc;
    TG=t_tg;
    T=t_tc;
}

void TestTensor::ToGPU(){
  Tensor::copy(TC,TG);
}


void TestTensor::check(string s) {
  Tensor::copy(TG,T);

  // equal(CPU,GPU)?
  int val=Tensor::equal(TC,T);

  if (!val) {
    cout<<"Fail "<<s<<"\n";
    exit(1);
  }
  else {
    cout<<"====================\n";
    cout<<"OK "<<s<<"\n";
    cout<<"====================\n";
  }
}


//////////////////////////////////////////
//////////// TEST MAIN ///////////////////
//////////////////////////////////////////
int main(int argc, char **argv) {
  int dim1,dim2,dim3;

  dim1=1000;
  dim2=1000;
  dim3=100;

  TestTensor *A=new TestTensor({dim1,dim3});
  TestTensor *B=new TestTensor({dim3,dim2});
  TestTensor *C=new TestTensor({dim1,dim2});

  TestTensor *Bt=new TestTensor({dim1,dim2});
  TestTensor *Bt2=new TestTensor({dim2,dim3});
  TestTensor *Ct2=new TestTensor({dim1,dim2});
  TestTensor *Ct=new TestTensor({dim3,dim2});

  TestTensor *D=new TestTensor({dim1,dim3});
  TestTensor *E=new TestTensor({dim1,dim3});

  TestTensor *F=new TestTensor({dim3});
  ////////////// COPY ////////////////////
  A->TC->rand_uniform(1.0);
  A->ToGPU();
  A->check("copy");

  ///////////// SET //////////////////////
  A->TC->set(1.0);
  A->TG->set(1.0);

  A->check("set");


    //////////// MAXPOOL /////////////////////
    // Test CPU
    float mpool_ref[16] = {12.0, 20.0, 30.0, 0.0, 8.0, 12.0, 2.0, 0.0, 34.0, 70.0, 37.0, 4.0, 112.0, 100.0, 25.0, 12.0};
    float mpool_sol[4] = {20.0, 30.0, 112.0, 37.0};

    // CPU input
    TestTensor *T_cpu=new TestTensor({1, 1, 4, 4});
    T_cpu->TC->ptr = mpool_ref;

    // GPU input
    TestTensor *T_gpu=new TestTensor({1, 1, 4, 4});
    T_gpu->TC->ptr = mpool_ref;
    T_gpu->ToGPU();

    // Result
    TestTensor *T_cpu_ref=new TestTensor({1, 1, 2, 2});
    T_cpu_ref->TC->ptr = mpool_sol;

    // [CPU] Instantiate PoolDescription + perform MaxPooling
    auto *pd_cpu = new PoolDescriptor(vector<int>{2,2}, vector<int>{2,2}, "none");
    pd_cpu->build(T_cpu->TC);
    pd_cpu->indX = new Tensor(pd_cpu->O->getShape(), DEV_CPU);
    pd_cpu->indY = new Tensor(pd_cpu->O->getShape(), DEV_CPU);
    Tensor::MPool2D(pd_cpu);

    // Check CPU correctness
    auto Z = new TestTensor(pd_cpu->O, T_cpu_ref->TC);
    Z->check("MPool2D CPU correctness");

//    printf("\nCPU result:\n");
//    pd_cpu->O->info();
//    pd_cpu->O->print();
//
//    printf("\nCPU correct solution:\n");
//    T_cpu_ref->TC->info();
//    T_cpu_ref->TC->print();

    // [GPU] Instantiate PoolDescription + perform MaxPooling
//    setbuf(stdout, NULL);
    auto *pd_gpu = new PoolDescriptor(vector<int>{2,2}, vector<int>{2,2}, "none");
    pd_gpu->build(T_gpu->TG);
    pd_gpu->indX = new Tensor(pd_gpu->O->getShape(), DEV_GPU);
    pd_gpu->indY = new Tensor(pd_gpu->O->getShape(), DEV_GPU);
    Tensor::MPool2D(pd_gpu);

    // Check GPU correctness
    auto Z2 = new TestTensor(pd_cpu->O, pd_gpu->O);
    Z2->check("MPool2D GPU correctness");

//    printf("\nGPU solution:\n");
//    pd_gpu->O->info();
//    pd_gpu->O->print();

  ///////////// total_sum ////////////////


  A->TC->rand_suniform(1);
  A->ToGPU();

  float fc= A->TC->sum();
  float fg= A->TG->sum();

  if (fabs(fc-fg)>0.01) {
    fprintf(stderr,"Fail total add %f!=%f\n",fc,fg);
    exit(EXIT_FAILURE);
  }
  cout<<"OK sum\n";


  //////////// MULT2D ///////////////////
  A->TC->rand_uniform(1.0);
  B->TC->rand_uniform(1.0);

  A->ToGPU();
  B->ToGPU();

  Tensor::mult2D(A->TC,0,B->TC,0,C->TC,0);
  Tensor::mult2D(A->TG,0,B->TG,0,C->TG,0);

  C->check("mult2D");

  A->TC->rand_uniform(1.0);
  Bt->TC->rand_uniform(1.0);

  A->ToGPU();
  Bt->ToGPU();

  Tensor::mult2D(A->TC,1,Bt->TC,0,Ct->TC,0);
  Tensor::mult2D(A->TG,1,Bt->TG,0,Ct->TG,0);

  Ct->check("mult2D Trasp");


  A->TC->rand_uniform(1.0);
  Bt2->TC->rand_uniform(1.0);

  A->ToGPU();
  Bt2->ToGPU();

  Tensor::mult2D(A->TC,0,Bt2->TC,1,Ct2->TC,0);
  Tensor::mult2D(A->TG,0,Bt2->TG,1,Ct2->TG,0);

  Ct2->check("mult2D Trasp2");


  A->TC->rand_uniform(1.0);
  Bt2->TC->rand_uniform(1.0);

  A->ToGPU();
  Bt2->ToGPU();

  Tensor::mult2D(A->TC,0,Bt2->TC,1,Ct2->TC,1);
  Tensor::mult2D(A->TG,0,Bt2->TG,1,Ct2->TG,1);

  Ct2->check("mult2D Trasp2 inc");


  //////////// SUM /////////////////////
  A->TC->rand_uniform(1.0);
  D->TC->rand_uniform(1.0);

  A->ToGPU();
  D->ToGPU();

  Tensor::sum(1.0,A->TC,1.0,D->TC,E->TC,0);
  Tensor::sum(1.0,A->TG,1.0,D->TG,E->TG,0);

  E->check("add");

  //////////// INC /////////////////////
  A->TC->rand_uniform(100.0);
  D->TC->rand_uniform(100.0);

  A->ToGPU();
  D->ToGPU();

  Tensor::inc(A->TC,D->TC);
  Tensor::inc(A->TG,D->TG);

  D->check("inc");


 //////////// Softmax /////////////////////
 A->TC->rand_suniform(100000);
 A->ToGPU();

 Tensor::Softmax(A->TC,D->TC);
 Tensor::Softmax(A->TG,D->TG);

 A->check("Softmax");

 //////////// Cross Ent ///////////////////
 A->TC->rand_uniform(1);
 D->TC->rand_binary(0.1);
 A->ToGPU();
 D->ToGPU();

 Tensor::cent(A->TC,D->TC,E->TC);
 Tensor::cent(A->TG,D->TG,E->TG);

 E->check("cross entropy");


 //////////// sum2D_rowwise ///////////////


 A->TC->rand_uniform(1.0);
 F->TC->rand_uniform(1.0);

 A->ToGPU();
 F->ToGPU();

 Tensor::sum2D_rowwise(A->TC, F->TC, D->TC);
 Tensor::sum2D_rowwise(A->TG, F->TG, D->TG);

 D->check("sum2D_rowwise");

 //////////// reduce_sum2D ////////////////
 A->TC->rand_uniform(1.0);
 F->TC->rand_uniform(1.0);

 A->ToGPU();
 F->ToGPU();

 Tensor::reduce_sum2D(A->TC, F->TC, 0, 0);
 Tensor::reduce_sum2D(A->TG, F->TG, 0, 0);

 F->check("reduce_sum2D");

 A->TC->rand_uniform(1.0);
 F->TC->rand_uniform(1.0);

 A->ToGPU();
 F->ToGPU();

 Tensor::reduce_sum2D(A->TC, F->TC, 0, 1);
 Tensor::reduce_sum2D(A->TG, F->TG, 0, 1);

 F->check("reduce_sum2D inc");

 //////////// ReLU ////////////////
 A->TC->rand_suniform(1.0);
 A->ToGPU();

 Tensor::ReLu(A->TC, D->TC);
 Tensor::ReLu(A->TG, D->TG);

 D->check("ReLU");

 //////////// D_ReLU ////////////////
 A->TC->rand_suniform(1.0);
 D->TC->rand_suniform(1.0);
 A->ToGPU();
 D->ToGPU();

 Tensor::D_ReLu(D->TC, A->TC,E->TC);
 Tensor::D_ReLu(D->TG, A->TG,E->TG);

 E->check("D_ReLU");



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
//    Tensor::reduce(A,B,axis,"add",false,NULL,0);
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
//    Tensor::reduce(A,B,axis2,"add",true,NULL,0);
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
