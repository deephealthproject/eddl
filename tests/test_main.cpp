/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <iostream>
#include "tensor/tensor.h"
#include "dev/aux_tests.h"

using namespace std;


// MPool data
float mpool_input[16] = {12.0, 20.0, 30.0, 0.0,
                         8.0, 12.0, 2.0, 0.0,
                         34.0, 70.0, 37.0, 4.0,
                         112.0, 100.0, 25.0, 12.0};
float mpool_sol[4] = {20.0, 30.0,
                      112.0, 37.0};
Tensor *t_mpool = new Tensor({1, 1, 4, 4}, mpool_input, DEV_CPU);
Tensor *t_mpool_sol = new Tensor({1, 1, 2, 2}, mpool_sol, DEV_CPU);


void pretty_res(string name, bool res){
    cout << "===================" << endl;
    cout << name << ": ";
    if(res){
        cout << "OK!";
    }else{
        cout << "FAILED!";
    }
    cout << endl;
    cout << "===================" << endl;
}


int main(int argc, char **argv) {
    TestResult res_cpu, res_gpu;

    // Check correctness
    pretty_res("MaxPool2D (CPU correctness)", check_tensors(run_mpool1(t_mpool, DEV_CPU).tensor, t_mpool_sol));
    pretty_res("MaxPool2D (GPU correctness)", check_tensors(run_mpool1(t_mpool, DEV_GPU).tensor, t_mpool_sol));
    pretty_res("MaxPool2D (CPU==GPU correctness)", check_tensors(run_mpool1(t_mpool, DEV_CPU).tensor, run_mpool1(t_mpool, DEV_GPU).tensor));
    
    // Check times
    res_cpu = run_mpool1(t_mpool, DEV_CPU, 1);
    pretty_res("MaxPool2D (CPU time=" + std::to_string(res_cpu.time)  + ")", true);
    res_gpu = run_mpool1(t_mpool, DEV_GPU, 1);
    pretty_res("MaxPool2D (GPU time=" + std::to_string(res_gpu.time)  + ")", true);
    pretty_res("MaxPool2D (GPU << CPU: x" + std::to_string(res_gpu.time/res_cpu.time) + ")", res_gpu.time<res_cpu.time);
}
