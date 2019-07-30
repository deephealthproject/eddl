
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

int main(int argc, char **argv) {

    get_fmem(pow(2, 20), "Super-small allocation"); // 4MB = 1MB * 4(float=4bytes)
    get_fmem(pow(2, 20)*100, "Small allocation"); // 400MB = 100MB * 4(float=4bytes)
    get_fmem(pow(2, 30), "Medium allocation"); // 4GB = 1GB * 4(float=4bytes)
    get_fmem(pow(2, 40), "Big allocation"); // 4TB = 1TB * 4(float=4bytes)

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
