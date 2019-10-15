/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <iostream>
#include <stdio.h>
#include <string>

#include "tensor/tensor.h"
#include "dev/aux_tests.h"

using namespace std;




void pretty_res(string text, bool res, string extra=""){
    cout << "===================" << endl;
    cout << text << ": ";
    if(res){
        cout << "OK!";
    }else{
        cout << "FAILED!";
    }
    cout << extra << endl;
    cout << "===================" << endl;
}


int main(int argc, char **argv) {
    TestResult res_small_cpu, res_small_gpu, res_big_cpu, res_big_gpu;

    // DATA *****************************************
    // MaxPool data
    float mpool_input[16] = {12.0, 20.0, 30.0, 0.0,
                             8.0, 12.0, 2.0, 0.0,
                             34.0, 70.0, 37.0, 4.0,
                             112.0, 100.0, 25.0, 12.0};
    float mpool_sol[4] = {20.0, 30.0,
                          112.0, 37.0};
    Tensor *t_mpool = new Tensor({1, 1, 4, 4}, mpool_input, DEV_CPU);
    Tensor *t_mpool_sol = new Tensor({1, 1, 2, 2}, mpool_sol, DEV_CPU);
    Tensor *t_mpool_big = Tensor::randn({1, 1, 1000, 1000}, DEV_CPU);


    // *** [MAXPOOL] *****************************************
    res_small_cpu = run_mpool1(t_mpool, DEV_CPU, 1);
    res_small_gpu = run_mpool1(t_mpool, DEV_GPU, 1);
    res_big_cpu = run_mpool1(t_mpool_big, DEV_CPU, 1);
    res_big_gpu = run_mpool1(t_mpool_big, DEV_GPU, 1);

    // Check correctness with solution
    pretty_res("MaxPool2D small (CPU correctness)", check_tensors(res_small_cpu.tensor, t_mpool_sol));
    pretty_res("MaxPool2D small (GPU correctness)", check_tensors(res_small_gpu.tensor, t_mpool_sol));
    pretty_res("MaxPool2D small (CPU==GPU correctness)", check_tensors(res_small_cpu.tensor, res_small_gpu.tensor));
    pretty_res("MaxPool2D big (CPU==GPU correctness)", check_tensors(res_big_cpu.tensor, res_big_gpu.tensor));
    pretty_res("MaxPool2D big (GPU << CPU: x" + std::to_string(res_big_cpu.time/res_big_gpu.time) + ")", res_big_gpu.time<res_big_cpu.time,
               "\n\t- CPU time: " + std::to_string(res_big_cpu.time) +
               "\n\t- GPU time: " + std::to_string(res_big_gpu.time));
}
