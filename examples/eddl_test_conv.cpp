
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>

#include "apis/eddl.h"
#include "utils.h"


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

void check_c_vs_g(Tensor *A, Tensor *B,string s)
{
  Tensor *C=new Tensor(B->getShape(),DEV_CPU);

  Tensor::copy(B,C);
  int val=Tensor::equal(A,C);

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

int main(int argc, char **argv) {


    TestTensor *A=new TestTensor({1,2,5,5});
    ConvolDescriptor *CDC=new ConvolDescriptor(vector<int>{3,3,3}, vector<int>{2,2}, vector<int>{1,1});
    ConvolDescriptor *CDG=new ConvolDescriptor(vector<int>{3,3,3}, vector<int>{2,2}, vector<int>{1,1});

    CDC->build(A->TC);
    CDG->build(A->TG);

    CDC->I->info();
    CDG->I->info();

    ////
    printf("FORW\n");

    CDC->I->rand_signed_uniform(0.1);
    CDC->K->rand_signed_uniform(0.1);
    CDC->bias->rand_signed_uniform(0.1);

    //CDC->I->set(1.0);
    //CDC->K->set(1.0);

    Tensor::copy(CDC->I,CDG->I);
    Tensor::copy(CDC->K,CDG->K);
    Tensor::copy(CDC->bias,CDG->bias);

    Conv2D(CDG);
    Conv2D(CDC);


    //CDC->O->print();
    //CDG->O->print();



    check_c_vs_g(CDC->O,CDG->O,"conv2d");


    ////
    printf("GRAD\n");

    CDC->D->rand_signed_uniform(0.1);
    CDC->gK->set(0.0);
    CDC->gbias->set(0.0);

    Tensor::copy(CDC->D,CDG->D);
    CDG->gK->set(0.0);
    CDG->gbias->set(0.0);

    Conv2D_grad(CDC);
    Conv2D_grad(CDG);

    check_c_vs_g(CDC->gK,CDG->gK,"conv2d_grad gK");
    check_c_vs_g(CDC->gbias,CDG->gbias,"conv2d_grad gbias");


    ////
    printf("BACK\n");
    CDC->ID=new Tensor(A->TC->getShape(),DEV_CPU);
    CDG->ID=new Tensor(A->TG->getShape(),DEV_GPU);

    CDC->D->rand_signed_uniform(0.1);
    CDC->ID->set(0.0);

    Tensor::copy(CDC->D,CDG->D);
    CDG->ID->set(0.0);


    Conv2D_back(CDC);
    Conv2D_back(CDG);

    check_c_vs_g(CDC->ID,CDG->ID,"conv2d_back");


}





























///
