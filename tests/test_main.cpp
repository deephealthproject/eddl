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


int total_ok = 0;
int total_errors = 0;

void pretty_res(string text, bool res, string extra=""){
    cout << "===================" << endl;
    cout << text << ": ";
    if(res){
        total_ok += 1;
        cout << "OK!";
    }else{
        total_errors += 1;
        cout << "FAILED!";
    }
    cout << extra << endl;
    cout << "===================" << endl;
}

void print_results_ref(string title, TestResult res_small_cpu, TestResult res_small_gpu, Tensor* t_sol){
        pretty_res(title + " small (CPU correctness)", check_tensors(res_small_cpu.tensor, t_sol));
        pretty_res(title + " small (GPU correctness)", check_tensors(res_small_gpu.tensor, t_sol));
        pretty_res(title + " small (CPU==GPU correctness)", check_tensors(res_small_cpu.tensor, res_small_gpu.tensor));
}

void print_results(string title, TestResult res_big_cpu, TestResult res_big_gpu){
    // Check correctness with solution
    pretty_res(title + " big (CPU==GPU correctness)", check_tensors(res_big_cpu.tensor, res_big_gpu.tensor));
    pretty_res(title + " big (GPU << CPU: x" + std::to_string(res_big_cpu.time/res_big_gpu.time) + ")", res_big_gpu.time<res_big_cpu.time,
               "\n\t- CPU time: " + std::to_string(res_big_cpu.time) +
               "\n\t- GPU time: " + std::to_string(res_big_gpu.time));
}

int main(int argc, char **argv) {
    TestResult res_small_cpu, res_small_gpu, res_big_cpu, res_big_gpu;
    Tensor *t_input, *t_input_sol, *t_weights, *t_input_big, *t_weights_big;
    string act;

    // *** [MAXPOOL] *****************************************
    // Data =================================
    float mpool_input[16] = {12.0, 20.0, 30.0, 0.0,
                             8.0, 12.0, 2.0, 0.0,
                             34.0, 70.0, 37.0, 4.0,
                             112.0, 100.0, 25.0, 12.0};
    float mpool_sol[4] = {20.0, 30.0,
                          112.0, 37.0};
    t_input = new Tensor({1, 1, 4, 4}, mpool_input, DEV_CPU);
    t_input_sol = new Tensor({1, 1, 2, 2}, mpool_sol, DEV_CPU);
    t_input_big = Tensor::randn({1, 1, 1000, 1000}, DEV_CPU);

    res_small_cpu = run_mpool(t_input, DEV_CPU, 1);
    res_small_gpu = run_mpool(t_input, DEV_GPU, 1);
    res_big_cpu = run_mpool(t_input_big, DEV_CPU, 1);
    res_big_gpu = run_mpool(t_input_big, DEV_GPU, 1);

    // Check correctness with solution
    print_results_ref("MaxPool2D", res_small_cpu, res_small_gpu, t_input_sol);
    print_results("MaxPool2D", res_big_cpu, res_big_gpu);


    // *** [CONV2D] *****************************************
    // Data =================================
    float conv2d_input[25] = {3.0, 3.0, 2.0, 1.0, 0.0,
                              0.0, 0.0, 1.0, 3.0, 1.0,
                              3.0, 1.0, 2.0, 2.0, 3.0,
                              2.0, 0.0, 0.0, 2.0, 2.0,
                              2.0, 0.0, 0.0, 0.0, 1.0};
    float conv2d_kernel[9] = {0.0, 1.0, 2.0,
                              2.0, 2.0, 0.0,
                              0.0, 1.0, 2.0};
    float conv2d_sol[9] = {12.0, 12.0, 17.0,
                           10.0, 17.0, 19.0,
                           9.0, 6.0, 14.0};
    t_input = new Tensor({1, 1, 5, 5}, conv2d_input, DEV_CPU);
    Tensor *t_conv2d_kernel = new Tensor({1, 1, 3, 3}, conv2d_kernel, DEV_CPU);
    t_input_sol = new Tensor({1, 1, 3, 3}, conv2d_sol, DEV_CPU);
    t_input_big =  Tensor::randn({1, 128, 100, 100}, DEV_CPU);
    Tensor *t_conv2d_kernel_big = Tensor::randn({128, 512, 3, 3}, DEV_CPU);


    res_small_cpu = run_conv2d(t_input, t_conv2d_kernel, DEV_CPU, 1);
    res_small_gpu = run_conv2d(t_input, t_conv2d_kernel, DEV_GPU, 1);
    res_big_cpu = run_conv2d(t_input_big, t_conv2d_kernel_big, DEV_CPU, 1);
    res_big_gpu = run_conv2d(t_input_big, t_conv2d_kernel_big, DEV_GPU, 1);

    // Check correctness with solution
    print_results_ref("Conv2D", res_small_cpu, res_small_gpu, t_input_sol);
    print_results("Conv2D", res_big_cpu, res_big_gpu);


    // *** [Dense] *****************************************
    // Data =================================
    int ndim = 3;
    int ndim_big = 1000*1000;
    t_input = Tensor::ones({1, 4}, DEV_CPU);
    t_weights = Tensor::full({4, ndim}, 2.0, DEV_CPU);
    t_input_sol = Tensor::full({1, 3}, 8.0, DEV_CPU);
    t_input_big = Tensor::ones({1, 4}, DEV_CPU);
    t_weights_big = Tensor::full({4, ndim_big}, 2.0, DEV_CPU);

    res_small_cpu = run_dense(t_input, t_weights, DEV_CPU, 1);
    res_small_gpu = run_dense(t_input, t_weights, DEV_GPU, 1);
    res_big_cpu = run_dense(t_input_big, t_weights_big, DEV_CPU, 1);
    res_big_gpu = run_dense(t_input_big, t_weights_big, DEV_GPU, 1);

    // Print results
    print_results_ref("Dense", res_small_cpu, res_small_gpu, t_input_sol);
    print_results("Dense", res_big_cpu, res_big_gpu);


    // *** [Sigmoid] *****************************************
    // Data =================================
    t_input_big = Tensor::randn({100, 10000}, DEV_CPU);
    act = "sigmoid";

    res_big_cpu = run_activation(t_input_big, act, DEV_CPU, 1);
    res_big_gpu = run_activation(t_input_big, act, DEV_GPU, 1);

    // Print results
    print_results(act, res_big_cpu, res_big_gpu);

    // *** [Relu] *****************************************
    // Data =================================
    t_input_big = Tensor::randn({100, 10000}, DEV_CPU);
    act = "relu";

    res_big_cpu = run_activation(t_input_big, act, DEV_CPU, 1);
    res_big_gpu = run_activation(t_input_big, act, DEV_GPU, 1);

    // Print results
    print_results(act, res_big_cpu, res_big_gpu);


    // *** [Softmax] *****************************************
    // Data =================================
    t_input_big = Tensor::randn({10000, 100}, DEV_CPU);
    act = "softmax";

    res_big_cpu = run_activation(t_input_big, act, DEV_CPU, 1);
    res_big_gpu = run_activation(t_input_big, act, DEV_GPU, 1);

    // Print results
    print_results(act, res_big_cpu, res_big_gpu);


    // *** [BathNorm] *****************************************
    // Data =================================
    t_input_big = Tensor::randn({1, 3, 1000, 1000}, DEV_CPU);

    res_big_cpu = run_batchnorm(t_input_big, DEV_CPU, 1);
    res_big_gpu = run_batchnorm(t_input_big, DEV_GPU, 1);

    // Print results
    print_results("BatchNorm", res_big_cpu, res_big_gpu);


    // *** [UpSampling] *****************************************
    // Data =================================
    vector<int> size = {2, 2};
    t_input_big = Tensor::randn({1000, 1000}, DEV_CPU);

    res_big_cpu = run_upsampling(t_input_big, size, DEV_CPU, 1);
    res_big_gpu = run_upsampling(t_input_big, size, DEV_GPU, 1);

    // Print results
    print_results("UpSampling", res_big_cpu, res_big_gpu);


    // *** [Math operations] *****************************************
    vector<string> math_ops = {"abs", "acos", "add", "asin", "atan", "ceil", "clamp", "cos", "cosh", "exp", "inv",
                               "floor", "log", "log2", "log10", "logn", "mod", "mult", "normalize", "pow", "reciprocal",
                               "remainder", "round", "rsqrt", "sigmoid", "sign", "sin", "sinh", "sqr", "sqrt", "tan",
                               "tanh", "trunc"};
    for (auto op:math_ops){
            t_input_big = Tensor::randn({1000, 1000}, DEV_CPU);
            res_big_cpu = run_tensor_op(t_input_big, op, DEV_CPU, 1);
            res_big_gpu = run_tensor_op(t_input_big, op, DEV_GPU, 1);
            print_results(op, res_big_cpu, res_big_gpu);
    }


    // *** [Summary] *****************************************
    float total_tests = total_ok + total_errors;
    cout << "" << endl;
    cout << "*************************************************" << endl;
    cout << "*** SUMMARY: ************************************" << endl;
    cout << "*************************************************" << endl;
    cout << "Total tests: " << total_ok + total_errors << endl;
    cout << "\t- Total ok: " << total_ok << " (" << (int)((float)total_ok/total_tests*100.0f) << "%)" << endl;
    cout << "\t- Total errors: " << total_errors << " (" << (int)((float)total_errors/total_tests*100.0f) << "%)" << endl;
    cout << "" << endl;

}
