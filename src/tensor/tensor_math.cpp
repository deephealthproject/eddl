/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cmath>
#include <limits>
#include <iostream>

#include "eddl/tensor/tensor.h"
#include "eddl/profiling.h"
#include "eddl/hardware/cpu/cpu_tensor.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_hw.h"
#endif

#ifdef cFPGA
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/fpga/nn/fpga_nn.h"
#endif

using namespace std;

// Math operations (Tensor-Tensor, Tensor-float) ************************

Tensor* Tensor::maximum(float v){
    Tensor *t = Tensor::empty_like(this);
    Tensor::maximum(this, t, v);
    return t;
}

Tensor* Tensor::maximum(Tensor* A, float v){
    Tensor *t = Tensor::empty_like(A);;
    Tensor::maximum(A, t, v);
    return t;
}

void Tensor::maximum(Tensor* A, Tensor* B, float v){

    PROFILING_HEADER_EXTERN(maximum);

    if (A->isCPU() && B->isCPU()){
        cpu_maximum(A, B, v);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_maximum(A, B, v);
    }
#endif
#ifdef cFPGA
    else {
      fpga_maximum(A, B, v);
    }
#endif

    PROFILING_FOOTER(maximum);
}

Tensor* Tensor::maximum(Tensor* A, Tensor* B){
    Tensor *t = Tensor::empty_like(A);;
    Tensor::maximum(A, B, t);
    return t;
}

void Tensor::maximum(Tensor* A, Tensor* B, Tensor* C){

    PROFILING_HEADER_EXTERN(maximum);

    if (A->isCPU() && B->isCPU() && C->isCPU()){
        cpu_maximum(A, B, C);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU() && C->isGPU())
    {
        gpu_maximum(A, B, C);
    }
#endif
#ifdef cFPGA
    else {
       fpga_maximum(A, B, C);
    }
#endif

    PROFILING_FOOTER(maximum);
}

Tensor* Tensor::minimum(float v){
    Tensor *t = Tensor::empty_like(this);
    Tensor::minimum(this, t, v);
    return t;
}

Tensor* Tensor::minimum(Tensor* A, float v){
    Tensor *t = Tensor::empty_like(A);;
    Tensor::minimum(A, t, v);
    return t;
}

void Tensor::minimum(Tensor* A, Tensor* B, float v){

    PROFILING_HEADER_EXTERN(minimum);

    if (A->isCPU() && B->isCPU()){
        cpu_minimum(A, B, v);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_minimum(A, B, v);
    }
#endif
#ifdef cFPGA
    else {
       fpga_minimum(A, B, v);
    }
#endif

    PROFILING_FOOTER(minimum);
}

Tensor* Tensor::minimum(Tensor* A, Tensor* B){
    Tensor *t = Tensor::empty_like(A);
    Tensor::minimum(A, B, t);
    return t;
}

void Tensor::minimum(Tensor* A, Tensor* B, Tensor* C){

    PROFILING_HEADER_EXTERN(minimum);

    if (A->isCPU() && B->isCPU() && C->isCPU()){
        cpu_minimum(A, B, C);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU() && C->isGPU())
    {
        gpu_minimum(A, B, C);
    }
#endif
#ifdef cFPGA
    else {
        fpga_minimum(A, B, C);
    }
#endif

    PROFILING_FOOTER(minimum);
}



// Math operations (reductions) ************************

float Tensor::max(){
    return Tensor::max(this);
}


float Tensor::max(Tensor* A){

    PROFILING_HEADER_EXTERN(max);

    if (A->isCPU()) {
        return cpu_max(A);
    }
#ifdef cGPU
    else if (A->isGPU())
    {
        return gpu_max(A);
    }
#endif
#ifdef cFPGA
    else if (A->isFPGA())
    {
        return fpga_max(A);
    }
#endif

    PROFILING_FOOTER(max);

    return 0.0f; // Never used, this is for the compiler warning
}

Tensor* Tensor::max(vector<int> axis, bool keepdims){
    // Build descriptor
    auto rd = new ReduceDescriptor2(axis, keepdims, this->device);
    rd->build(this->shape);

    // Create output tensor
    Tensor *t = Tensor::empty(rd->oshape, this->device);
    Tensor::max(this, t, rd);

    delete rd;
    return t;
}

void Tensor::max(Tensor* A, Tensor *B, ReduceDescriptor2 *rd){

    PROFILING_HEADER_EXTERN(max);

    if (A->isCPU() && B->isCPU()) {
        cpu_max(A, B, rd);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_max(A, B, rd);
    }
#endif
#ifdef cFPGA
    else {
        fpga_max(A, B, rd);
    }
#endif

    PROFILING_FOOTER(max);
}



int Tensor::argmax(){
    return Tensor::argmax(this);
}


int Tensor::argmax(Tensor* A){

    PROFILING_HEADER_EXTERN(argmax);

    if (A->isCPU()) {
        PROFILING_FOOTER(argmax);
        return cpu_argmax(A);
    }
#ifdef cGPU
    else if (A->isGPU())
    {
        PROFILING_FOOTER(argmax);
        return gpu_argmax(A);
    }
#endif
#ifdef cFPGA
    else {
        PROFILING_FOOTER(argmax);
        return fpga_argmax(A);
    }
#endif

    PROFILING_FOOTER(argmax);

    msg("Invalid device", "Tensor::argmax");
    return 0.0f; // Never used, this is for the compiler warning
}

Tensor* Tensor::argmax(vector<int> axis, bool keepdims){
    // Build descriptor
    auto rd = new ReduceDescriptor2(axis, keepdims, this->device);
    rd->build(this->shape);

    // Create output tensor
    Tensor *t = Tensor::empty(rd->oshape, this->device);
    Tensor::argmax(this, t, rd);

    delete rd;
    return t;
}

void Tensor::argmax(Tensor* A, Tensor *B, ReduceDescriptor2 *rd){

    PROFILING_HEADER_EXTERN(argmax);

    if (A->isCPU() && B->isCPU()) {
        cpu_argmax(A, B, rd);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_argmax(A, B, rd);
    }
#endif
#ifdef cFPGA
    else {
        fpga_argmax(A, B, rd);
    }
#endif

    PROFILING_FOOTER(argmax);
}

void Tensor::argmax_d(Tensor *D, Tensor *O, Tensor *PD){

    PROFILING_HEADER_EXTERN(argmax_d);

    if (D->isCPU() && O->isCPU() && PD->isCPU()) {
        cpu_argmax_d(D, O, PD);
    }
#ifdef cGPU
    else if (D->isGPU() && O->isGPU() && PD->isGPU())
    {
        gpu_argmax_d(D, O, PD);
    }
#endif
#ifdef cFPGA
    else {
        //fpga_argmax_d(D, O, PD);
    }
#endif

    PROFILING_FOOTER(argmax_d);
}

float Tensor::min(){
    return Tensor::min(this);
}


float Tensor::min(Tensor* A){

    PROFILING_HEADER_EXTERN(min);

    if (A->isCPU()) {
        PROFILING_FOOTER(min);
        return cpu_min(A);
    }
#ifdef cGPU
    else if (A->isGPU())
    {
        PROFILING_FOOTER(min);
	return gpu_min(A);
    }
#endif
#ifdef cFPGA
    else {
        PROFILING_FOOTER(min);
        return fpga_min(A);
    }
#endif

    PROFILING_FOOTER(min);

    msg("Invalid device", "Tensor::min");
    return 0.0f; // Never used, this is for the compiler warning
}

Tensor* Tensor::min(vector<int> axis, bool keepdims){
    // Build descriptor
    auto rd = new ReduceDescriptor2(axis, keepdims, this->device);
    rd->build(this->shape);

    // Create output tensor
    Tensor *t = Tensor::empty(rd->oshape, this->device);
    Tensor::min(this, t, rd);

    delete rd;
    return t;
}

void Tensor::min(Tensor* A, Tensor *B, ReduceDescriptor2 *rd){

    PROFILING_HEADER_EXTERN(min);

    if (A->isCPU() && B->isCPU()) {
        cpu_min(A, B, rd);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_min(A, B, rd);
    }
#endif
#ifdef cFPGA
    else {
        fpga_min(A, B, rd);
    }
#endif

    PROFILING_FOOTER(min);
}


int Tensor::argmin(){
    return Tensor::argmin(this);
}


int Tensor::argmin(Tensor* A){

    PROFILING_HEADER_EXTERN(argmin);

    if (A->isCPU()) {
        PROFILING_FOOTER(argmin);
        return cpu_argmin(A);
    }
#ifdef cGPU
    else if (A->isGPU())
    {
        PROFILING_FOOTER(argmin);
        return gpu_argmin(A);
    }
#endif
#ifdef cFPGA
    else {
        PROFILING_FOOTER(argmin);
        return fpga_argmin(A);
    }
#endif

    PROFILING_FOOTER(argmin);

    msg("Invalid device", "Tensor::argmax");
    return 0.0f; // Never used, this is for the compiler warning
}

Tensor* Tensor::argmin(vector<int> axis, bool keepdims){
    // Build descriptor
    auto rd = new ReduceDescriptor2(axis, keepdims, this->device);
    rd->build(this->shape);

    // Create output tensor
    Tensor *t = Tensor::empty(rd->oshape, this->device);
    Tensor::argmin(this, t, rd);

    delete rd;
    return t;
}

void Tensor::argmin(Tensor* A, Tensor *B, ReduceDescriptor2 *rd){

    PROFILING_HEADER_EXTERN(argmin);

    if (A->isCPU() && B->isCPU()) {
        cpu_argmin(A, B, rd);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_argmin(A, B, rd);
    }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
    {
        fpga_argmin(A, B, rd);
    }
#endif

    PROFILING_FOOTER(argmin);
}


float Tensor::sum(){
    return Tensor::sum(this);
}


float Tensor::sum(Tensor* A){

    PROFILING_HEADER_EXTERN(sum);

    if (A->isCPU()) {
        PROFILING_FOOTER(sum);
        return cpu_sum(A);
    }
#ifdef cGPU
    else if (A->isGPU())
    {

        PROFILING_FOOTER(sum);
        return gpu_sum(A);
    }
#endif
#ifdef cFPGA
    else {

        PROFILING_FOOTER(sum);
        return fpga_sum(A);
    }
#endif

    PROFILING_FOOTER(sum);

    msg("Invalid device", "Tensor::sum");
    return 0.0f; // Never used, this is for the compiler warning
}

Tensor* Tensor::sum(vector<int> axis, bool keepdims){
    // Build descriptor
    auto rd = new ReduceDescriptor2(axis, keepdims, this->device);
    rd->build(this->shape);

    // Create output tensor
    Tensor *t = Tensor::empty(rd->oshape, this->device);
    Tensor::sum(this, t, rd);

    delete rd;
    return t;
}

void Tensor::sum(Tensor* A, Tensor *B, ReduceDescriptor2 *rd){

    PROFILING_HEADER_EXTERN(sum);

    if (A->isCPU() && B->isCPU()) {
        cpu_sum(A, B, rd);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
      gpu_sum(A, B, rd);
    }
#endif
#ifdef cFPGA
    else {
        fpga_sum(A, B, rd);
    }
#endif

    PROFILING_FOOTER(sum);
}

float Tensor::sum_abs(){
    return Tensor::sum_abs(this);
}


float Tensor::sum_abs(Tensor* A){

    PROFILING_HEADER_EXTERN(sum_abs);

    if (A->isCPU()) {

        PROFILING_FOOTER(sum_abs);
	return cpu_sum_abs(A);
    }
#ifdef cGPU
    else if (A->isGPU())
    {
        PROFILING_FOOTER(sum_abs);
        return gpu_sum_abs(A);
    }
#endif
#ifdef cFPGA
    else {
        PROFILING_FOOTER(sum_abs);
        return fpga_sum_abs(A);
    }
#endif

    PROFILING_FOOTER(sum_abs);

    msg("Invalid device", "Tensor::sum_abs");
    return 0.0f; // Never used, this is for the compiler warning
}


Tensor* Tensor::sum_abs(vector<int> axis, bool keepdims){
    // Build descriptor
    auto rd = new ReduceDescriptor2(axis, keepdims, this->device);
    rd->build(this->shape);

    // Create output tensor
    Tensor *t = Tensor::empty(rd->oshape, this->device);
    Tensor::sum_abs(this, t, rd);

    delete rd;
    return t;
}

void Tensor::sum_abs(Tensor* A, Tensor *B, ReduceDescriptor2 *rd){

    PROFILING_HEADER_EXTERN(sum_abs);

    if (A->isCPU() && B->isCPU()) {
        cpu_sum_abs(A, B, rd);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_sum_abs(A, B, rd);
    }
#endif
#ifdef cFPGA
    else {
        fpga_sum_abs(A, B, rd);
    }
#endif

    PROFILING_FOOTER(sum_abs);
}

float Tensor::prod(){
    return Tensor::prod(this);
}


float Tensor::prod(Tensor* A){  // AKA factorial

    PROFILING_HEADER_EXTERN(prod);

    if (A->isCPU()) {
        PROFILING_FOOTER(prod);
        return cpu_prod(A);
    }
#ifdef cGPU
    else if (A->isGPU())
    {
        PROFILING_FOOTER(prod);
        return gpu_prod(A);
    }
#endif
#ifdef cFPGA
    else {
        PROFILING_FOOTER(prod);
        return fpga_prod(A);
    }
#endif

    PROFILING_FOOTER(prod);

    msg("Invalid device", "Tensor::prod");
    return 0.0f; // Never used, this is for the compiler warning
}


Tensor* Tensor::prod(vector<int> axis, bool keepdims){
    // Build descriptor
    auto rd = new ReduceDescriptor2(axis, keepdims, this->device);
    rd->build(this->shape);

    // Create output tensor
    Tensor *t = Tensor::empty(rd->oshape, this->device);
    Tensor::prod(this, t, rd);

    delete rd;
    return t;
}

void Tensor::prod(Tensor* A, Tensor *B, ReduceDescriptor2 *rd){

    PROFILING_HEADER_EXTERN(prod);

    if (A->isCPU() && B->isCPU()) {
        cpu_prod(A, B, rd);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_prod(A, B, rd);
    }
#endif
#ifdef cFPGA
    else {
        fpga_prod(A, B, rd);
    }
#endif

    PROFILING_FOOTER(prod);
}


float Tensor::mean(){
    return Tensor::mean(this);
}

float Tensor::mean(Tensor* A){
    float sum = A->sum();
    return sum/A->size;
}

Tensor* Tensor::mean(vector<int> axis, bool keepdims){
    // Build descriptor
    auto rd = new ReduceDescriptor2(axis, keepdims, this->device);
    rd->build(this->shape);

    // Create output tensor
    Tensor *t = Tensor::empty(rd->oshape, this->device);
    Tensor::mean(this, t, rd);

    delete rd;
    return t;
}

void Tensor::mean(Tensor* A, Tensor *B, ReduceDescriptor2 *rd){

    PROFILING_HEADER_EXTERN(mean);

    if (A->isCPU() && B->isCPU()) {
        cpu_mean(A, B, rd);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_mean(A, B, rd);
    }
#endif
#ifdef cFPGA
    else {
        fpga_mean(A, B, rd);
    }
#endif

    PROFILING_FOOTER(mean);
}



float Tensor::median(){
    return Tensor::median(this);
}


float Tensor::median(Tensor* A){

    PROFILING_HEADER_EXTERN(median);

    float res = 0.0f;

    // Clone tensor (needs to be sorted first)
    Tensor *tmp = A->clone();
    tmp->sort_();

    if (tmp->isCPU()) {
        res = cpu_median(tmp);
    }
#ifdef cGPU
    else if (tmp->isGPU()) {
        res = gpu_median(tmp);
    }
#endif
#ifdef cFPGA
    else {
        res = fpga_median(tmp);
    }
#endif

    PROFILING_FOOTER(median);

    delete tmp;
    return res;
}

Tensor* Tensor::median(vector<int> axis, bool keepdims){
    // Build descriptor
    auto rd = new ReduceDescriptor2(axis, keepdims, this->device);
    rd->build(this->shape);

    // Create output tensor
    Tensor *t = Tensor::empty(rd->oshape, this->device);
    Tensor::median(this, t, rd);

    delete rd;
    return t;
}

void Tensor::median(Tensor* A, Tensor *B, ReduceDescriptor2 *rd){

    PROFILING_HEADER_EXTERN(median);

    if (A->isCPU() && B->isCPU()) {
        cpu_median(A, B, rd);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_median(A, B, rd);
    }
#endif
#ifdef cFPGA
    else {
        fpga_median(A, B, rd);
    }
#endif

    PROFILING_FOOTER(median);
}

float Tensor::std(bool unbiased){
    return Tensor::std(this, unbiased);
}


float Tensor::std(Tensor* A, bool unbiased){

    PROFILING_HEADER_EXTERN(std);

    if (A->isCPU()) {
        PROFILING_FOOTER(std);
        return cpu_std(A, unbiased);
    }
#ifdef cGPU
    else if (A->isGPU())
    {
        PROFILING_FOOTER(std);
        return gpu_std(A, unbiased);
    }
#endif
#ifdef cFPGA
    else {
        PROFILING_FOOTER(std);
        fpga_std(A, unbiased);
    }
#endif

    PROFILING_FOOTER(std);

    msg("Invalid device", "Tensor::std");
    return 0.0f; // Never used, this is for the compiler warning
}


Tensor* Tensor::std(vector<int> axis, bool keepdims, bool unbiased){
    // Build descriptor
    auto rd = new ReduceDescriptor2(axis, keepdims, this->device);
    rd->build(this->shape);

    // Create output tensor
    Tensor *t = Tensor::empty(rd->oshape, this->device);
    Tensor::std(this, t, rd, unbiased);

    delete rd;
    return t;
}

void Tensor::std(Tensor* A, Tensor *B, ReduceDescriptor2 *rd, bool unbiased){

    PROFILING_HEADER_EXTERN(std);

    if (A->isCPU() && B->isCPU()) {
        cpu_std(A, B, rd, unbiased);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_std(A, B, rd, unbiased);
    }
#endif
#ifdef cFPGA
    else {
        fpga_std(A, B, rd, unbiased);
    }
#endif

    PROFILING_FOOTER(std);
}


float Tensor::var(bool unbiased){
    return Tensor::var(this, unbiased);
}


float Tensor::var(Tensor* A, bool unbiased){

    PROFILING_HEADER_EXTERN(var);

    if (A->isCPU()) {

        PROFILING_FOOTER(var);
        return cpu_var(A, unbiased);
    }
#ifdef cGPU
    else if (A->isGPU())
    {

        PROFILING_FOOTER(var);
        return gpu_var(A, unbiased);
    }
#endif
#ifdef cFPGA
    else if (A->isFPGA())
    {

        PROFILING_FOOTER(var);
        return fpga_var(A, unbiased);
    }
#endif

    PROFILING_FOOTER(var);

    msg("Invalid device", "Tensor::var");
    return 0.0f; // Never used, this is for the compiler warning
}


Tensor* Tensor::var(vector<int> axis, bool keepdims, bool unbiased){
    // Build descriptor
    auto rd = new ReduceDescriptor2(axis, keepdims, this->device);
    rd->build(this->shape);

    // Create output tensor
    Tensor *t = Tensor::empty(rd->oshape, this->device);
    Tensor::var(this, t, rd, unbiased);

    delete rd;
    return t;
}

void Tensor::var(Tensor* A, Tensor *B, ReduceDescriptor2 *rd, bool unbiased){

    PROFILING_HEADER_EXTERN(var);

    if (A->isCPU() && B->isCPU()) {
        cpu_var(A, B, rd, unbiased);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_var(A, B, rd, unbiased);
    }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
    {
        fpga_var(A, B, rd, unbiased);
    }
#endif

    PROFILING_FOOTER(var);
}


int Tensor::mode(){
    return Tensor::mode(this);
}


int Tensor::mode(Tensor* A){

    PROFILING_HEADER_EXTERN(mode);

    if (A->isCPU()) {
	    PROFILING_FOOTER(mode);
        return cpu_mode(A);
    }
#ifdef cGPU
    else if (A->isGPU())
    {
	    PROFILING_FOOTER(mode);
        return gpu_mode(A);
    }
#endif
#ifdef cFPGA
    else if (A->isFPGA())
    {
	    PROFILING_FOOTER(mode);
        return fpga_mode(A);
    }
#endif

    PROFILING_FOOTER(mode);

    msg("Invalid device", "Tensor::mode");
    return 0; // Never used, this is for the compiler warning
}


Tensor* Tensor::mode(vector<int> axis, bool keepdims){
    // Build descriptor
    auto rd = new ReduceDescriptor2(axis, keepdims, this->device);
    rd->build(this->shape);

    // Create output tensor
    Tensor *t = Tensor::empty(rd->oshape, this->device);
    Tensor::mode(this, t, rd);

    delete rd;
    return t;
}

void Tensor::mode(Tensor* A, Tensor *B, ReduceDescriptor2 *rd){

    PROFILING_HEADER_EXTERN(mode);

    if (A->isCPU() && B->isCPU()) {
        cpu_mode(A, B, rd);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_mode(A, B, rd);
    }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
    {
        fpga_mode(A, B, rd);
    }
#endif

    PROFILING_FOOTER(mode);
}


void Tensor::abs_(){
    Tensor::abs(this, this);
}

Tensor* Tensor::abs(){
    Tensor *t = this->clone();
    t->abs_();
    return t;
}

void Tensor::abs(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(abs);

    if (A->isCPU() && B->isCPU()) {
        cpu_abs(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_abs(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_abs(A, B);
      }
#endif

    PROFILING_FOOTER(abs);
}


void Tensor::acos_(){
    Tensor::acos(this, this);
}

Tensor* Tensor::acos(){
    Tensor *t = this->clone();
    t->acos_();
    return t;
}

void Tensor::acos(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(acos);

    if (A->isCPU() && B->isCPU()) {
        cpu_acos(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_acos(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_acos(A, B);
      }
#endif

    PROFILING_FOOTER(acos);
}


void Tensor::add_(float v){
    Tensor::add(this, this, v);
}

Tensor* Tensor::add(float v){
    Tensor *t = this->clone();
    t->add_(v);
    return t;
}

void Tensor::add_(Tensor* A){
    Tensor::add(this, A, this);
}

Tensor* Tensor::add(Tensor* A){
    Tensor *t = this->clone();
    t->add_(A);
    return t;
}

void Tensor::add(Tensor *A, Tensor *B, float v){
    PROFILING_HEADER_EXTERN(add);

    if (A->isCPU() && B->isCPU()) {
        cpu_add(A, B, v);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_add(A, B, v);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_add(A, B, v);
      }
#endif

    PROFILING_FOOTER(add);
}


void Tensor::asin_(){
    Tensor::asin(this, this);
}


Tensor* Tensor::asin(){
    Tensor *t = this->clone();
    t->asin_();
    return t;
}

void Tensor::asin(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(asin);

    if (A->isCPU() && B->isCPU()) {
        cpu_asin(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_asin(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_asin(A, B);
      }
#endif

    PROFILING_FOOTER(asin); 
}


void Tensor::atan_(){
    Tensor::atan(this, this);
}


Tensor* Tensor::atan(){
    Tensor *t = this->clone();
    t->atan_();
    return t;
}


void Tensor::atan(Tensor *A, Tensor *B){
    
    PROFILING_HEADER_EXTERN(atan);    

    if (A->isCPU() && B->isCPU()) {
        cpu_atan(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_atan(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_atan(A, B);
      }
#endif

    PROFILING_FOOTER(atan);
}


void Tensor::ceil_(){
    Tensor::ceil(this, this);
}


Tensor* Tensor::ceil(){
    Tensor *t = this->clone();
    t->ceil_();
    return t;
}


void Tensor::ceil(Tensor *A, Tensor *B){
    
    PROFILING_HEADER_EXTERN(ceil);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_ceil(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_ceil(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_ceil(A, B);
      }
#endif

    PROFILING_FOOTER(ceil);
}


void Tensor::clamp_(float min, float max){
    Tensor::clamp(this, this, min, max);
}


Tensor* Tensor::clamp(float min, float max){
    Tensor *t = this->clone();
    t->clamp_(min, max);
    return t;
}


void Tensor::clamp(Tensor *A, Tensor *B, float min, float max){

    PROFILING_HEADER_EXTERN(clamp);

    if (A->isCPU() && B->isCPU()) {
        cpu_clamp(A, B, min, max);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_clamp(A, B, min, max);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_clamp(A, B, min, max);
      }
#endif

    PROFILING_FOOTER(clamp);
}


void Tensor::clampmax_(float max){
    Tensor::clampmax(this, this, max);
}


Tensor* Tensor::clampmax(float max){
    Tensor *t = this->clone();
    t->clampmax_(max);
    return t;
}

void Tensor::clampmax(Tensor *A, Tensor *B, float max){
    Tensor::clamp(A, B, MIN_FLOAT, max);
}


void Tensor::clampmin_(float min){
    Tensor::clampmin(this, this, min);
}


Tensor* Tensor::clampmin(float min){
    Tensor *t = this->clone();
    t->clampmin_(min);
    return t;
}


void Tensor::clampmin(Tensor *A, Tensor *B, float min){
    Tensor::clamp(A, B, min, MAX_FLOAT);
}


void Tensor::cos_(){
    Tensor::cos(this, this);
}


Tensor* Tensor::cos(){
    Tensor *t = this->clone();
    t->cos_();
    return t;
}


void Tensor::cos(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(cos);

    if (A->isCPU() && B->isCPU()) {
        cpu_cos(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_cos(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_cos(A, B);
      }
#endif

    PROFILING_FOOTER(cos);
}


void Tensor::cosh_(){
    Tensor::cosh(this, this);
}


Tensor* Tensor::cosh(){
    Tensor *t = this->clone();
    t->cosh_();
    return t;
}

void Tensor::cosh(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(cosh);

    if (A->isCPU() && B->isCPU()) {
        cpu_cosh(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_cosh(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_cosh(A, B);
      }
#endif

    PROFILING_FOOTER(cosh);
}


void Tensor::div_(float v){
    Tensor::div(this, this, v);
}


Tensor* Tensor::div(float v){
    Tensor *t = this->clone();
    t->div_(v);
    return t;
}


void Tensor::div_(Tensor* A){
    Tensor::div(this, A, this);
}


Tensor* Tensor::div(Tensor* A){
    Tensor *t = this->clone();
    t->div_(A);
    return t;
}


void Tensor::div(Tensor *A, Tensor *B, float v){
    Tensor::mult(A, B, 1.0f/v);
}


void Tensor::exp_(){
    Tensor::exp(this, this);
}


Tensor* Tensor::exp(){
    Tensor *t = this->clone();
    t->exp_();
    return t;
}


void Tensor::exp(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(exp);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_exp(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_exp(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_exp(A, B);
      }
#endif

    PROFILING_FOOTER(exp);
}


void Tensor::floor_(){
    Tensor::floor(this, this);
}


Tensor* Tensor::floor(){
    Tensor *t = this->clone();
    t->floor_();
    return t;
}


void Tensor::floor(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(floor);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_floor(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_floor(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_floor(A, B);
      }
#endif

    PROFILING_FOOTER(floor);
}


void Tensor::inv_(float v){
    Tensor::inv(this, this, v);
}


Tensor* Tensor::inv(float v){
    Tensor *t = this->clone();
    t->inv_(v);
    return t;
}


void Tensor::inv(Tensor *A, Tensor *B, float v){

    PROFILING_HEADER_EXTERN(inv);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_inv(A, B, v);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_inv(A, B, v);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_inv(A, B, v);
      }
#endif

    PROFILING_FOOTER(inv);   
}


void Tensor::log_(){
    Tensor::log(this, this);
}


Tensor* Tensor::log(){
    Tensor *t = this->clone();
    t->log_();
    return t;
}


void Tensor::log(Tensor *A, Tensor *B){
    
    PROFILING_HEADER_EXTERN(log);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_log(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_log(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_log(A, B);
      }
#endif

    PROFILING_FOOTER(log);
}


void Tensor::log2_(){
    Tensor::log2(this, this);
}


Tensor* Tensor::log2(){
    Tensor *t = this->clone();
    t->log2_();
    return t;
}


void Tensor::log2(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(log2);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_log2(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_log2(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_log2(A, B);
      }
#endif

    PROFILING_FOOTER(log2);
}


void Tensor::log10_(){
    Tensor::log10(this, this);
}


Tensor* Tensor::log10(){
    Tensor *t = this->clone();
    t->log10_();
    return t;
}


void Tensor::log10(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(log10);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_log10(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_log10(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_log10(A, B);
      }
#endif

    PROFILING_FOOTER(log10);
}


void Tensor::logn_(float n){
    Tensor::logn(this, this, n);
}


Tensor* Tensor::logn(float n){
    Tensor *t = this->clone();
    t->logn_(n);
    return t;
}


void Tensor::logn(Tensor *A, Tensor *B, float n){
    
    PROFILING_HEADER_EXTERN(logn);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_logn(A, B, n);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_logn(A, B, n);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_logn(A, B, n);
      }
#endif

    PROFILING_FOOTER(logn);
}


void Tensor::mod_(float v){
    Tensor::mod(this, this, v);
}


Tensor* Tensor::mod(float v){
    Tensor *t = this->clone();
    t->mod_(v);
    return t;
}


void Tensor::mod(Tensor *A, Tensor *B, float v){

    PROFILING_HEADER_EXTERN(mod);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_mod(A, B, v);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_mod(A, B, v);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_mod(A, B, v);
      }
#endif

    PROFILING_FOOTER(mod);
}


void Tensor::mult_(float v){
    Tensor::mult(this, this, v);
}


Tensor* Tensor::mult(float v){
    Tensor *t = this->clone();
    t->mult_(v);
    return t;
}


void Tensor::mult_(Tensor* A){
    Tensor::mult(this, A, this);
}


Tensor* Tensor::mult(Tensor* A){
    Tensor *t = this->clone();
    t->mult_(A);
    return t;
}


void Tensor::mult(Tensor *A, Tensor *B, float v){

    PROFILING_HEADER_EXTERN(mult);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_mult(A, B, v);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_mult(A, B, v);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_mult(A, B, v);
      }
#endif

    PROFILING_FOOTER(mult);
}


void Tensor::neg_(){
    Tensor::neg(this, this);
}


Tensor* Tensor::neg(){
    Tensor *t = this->clone();
    t->neg_();
    return t;
}


void Tensor::neg(Tensor *A, Tensor *B){
    Tensor::mult(A, B, -1.0f);
}


void Tensor::normalize_(float min, float max){
    Tensor::normalize(this, this, min, max);
}


Tensor* Tensor::normalize(float min, float max){
    Tensor *t = this->clone();
    t->normalize_(min, max);
    return t;
}


void Tensor::normalize(Tensor *A, Tensor *B, float min, float max){

    PROFILING_HEADER_EXTERN(normalize);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_normalize(A, B, min, max);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_normalize(A, B, min, max);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_normalize(A, B, min, max);
      }
#endif

    PROFILING_FOOTER(normalize);
}


void Tensor::pow_(float exp){
    Tensor::pow(this, this, exp);
}


Tensor* Tensor::pow(float exp){
    Tensor *t = this->clone();
    t->pow_(exp);
    return t;
}


void Tensor::pow(Tensor *A, Tensor *B, float exp){
    
    PROFILING_HEADER_EXTERN(pow);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_pow(A, B, exp);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_pow(A, B, exp);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_pow(A, B, exp);
      }
#endif

    PROFILING_FOOTER(pow);
}


void Tensor::powb_(float base){
    Tensor::powb(this, this, base);
}


Tensor* Tensor::powb(float base){
    Tensor *t = this->clone();
    t->powb_(base);
    return t;
}


void Tensor::powb(Tensor *A, Tensor *B, float base){

    PROFILING_HEADER_EXTERN(powb);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_powb(A, B, base);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_powb(A, B, base);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_powb(A, B, base);
      }
#endif

    PROFILING_FOOTER(powb);
}


void Tensor::reciprocal_(){
    Tensor::reciprocal(this, this);
}


Tensor* Tensor::reciprocal(){
    Tensor *t = this->clone();
    t->reciprocal_();
    return t;
}


void Tensor::reciprocal(Tensor *A, Tensor *B){
    Tensor::inv(A, B, 1.0f);
}


void Tensor::remainder_(float v){
    Tensor::remainder(this, this, v);
}


Tensor* Tensor::remainder(float v){
    Tensor *t = this->clone();
    t->remainder_(v);
    return t;
}


void Tensor::remainder(Tensor *A, Tensor *B, float v){

    PROFILING_HEADER_EXTERN(remainder);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_remainder(A, B, v);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_remainder(A, B, v);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_remainder(A, B, v);
      }
#endif


    PROFILING_FOOTER(remainder);
}


void Tensor::round_(){
    Tensor::round(this, this);
}


Tensor* Tensor::round(){
    Tensor *t = this->clone();
    t->round_();
    return t;
}


void Tensor::round(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(round);

    if (A->isCPU() && B->isCPU()) {
        cpu_round(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_round(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_round(A, B);
      }
#endif

    PROFILING_FOOTER(round);
}


void Tensor::rsqrt_(){
    Tensor::rsqrt(this, this);
}


Tensor* Tensor::rsqrt(){
    Tensor *t = this->clone();
    t->rsqrt_();
    return t;
}


void Tensor::rsqrt(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(rsqrt);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_rsqrt(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_rsqrt(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_rsqrt(A, B);
      }
#endif

    PROFILING_FOOTER(rsqrt);
}


void Tensor::sigmoid_(){
    Tensor::sigmoid(this, this);
}


Tensor* Tensor::sigmoid(){
    Tensor *t = this->clone();
    t->sigmoid_();
    return t;
}


void Tensor::sigmoid(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(sigmoid);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_sigmoid(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_sigmoid(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_sigmoid(A, B);
      }
#endif

    PROFILING_FOOTER(sigmoid);
}


void Tensor::sign_(float zero_sign){
    Tensor::sign(this, this, zero_sign);
}


Tensor* Tensor::sign(float zero_sign){
    Tensor *t = this->clone();
    t->sign_(zero_sign);
    return t;
}


void Tensor::sign(Tensor *A, Tensor *B, float zero_sign) {

    PROFILING_HEADER_EXTERN(sign);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_sign(A, B, zero_sign);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_sign(A, B, zero_sign);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_sign(A, B, zero_sign);
      }
#endif

    PROFILING_FOOTER(sign);
}


void Tensor::sin_(){
    Tensor::sin(this, this);
}


Tensor* Tensor::sin(){
    Tensor *t = this->clone();
    t->sin_();
    return t;
}


void Tensor::sin(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(sin);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_sin(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_sin(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_sin(A, B);
      }
#endif

    PROFILING_FOOTER(sin);
}


void Tensor::sinh_(){
    Tensor::sinh(this, this);
}


Tensor* Tensor::sinh(){
    Tensor *t = this->clone();
    t->sinh_();
    return t;
}


void Tensor::sinh(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(sinh);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_sinh(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_sinh(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_sinh(A, B);
      }
#endif

    PROFILING_FOOTER(sinh);
}


void Tensor::sqr_(){
    Tensor::sqr(this, this);
}


Tensor* Tensor::sqr(){
    Tensor *t = this->clone();
    t->sqr_();
    return t;
}


void Tensor::sqr(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(sqr);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_sqr(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_sqr(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_sqr(A, B);
      }
#endif

    PROFILING_FOOTER(sqr);
}


void Tensor::sqrt_(){
    Tensor::sqrt(this, this);
}


Tensor* Tensor::sqrt(){
    Tensor *t = this->clone();
    t->sqrt_();
    return t;
}


void Tensor::sqrt(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(sqrt);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_sqrt(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_sqrt(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_sqrt(A, B);
      }
#endif

    PROFILING_FOOTER(sqrt);
}


void Tensor::sub_(float v){
    Tensor::sub(this, this, v);
}


Tensor* Tensor::sub(float v){
    Tensor *t = this->clone();
    t->sub_(v);
    return t;
}


void Tensor::sub_(Tensor* A){
    Tensor::sub(this, A, this);
}


Tensor* Tensor::sub(Tensor* A){
    Tensor *t = this->clone();
    t->sub_(A);
    return t;
}


void Tensor::sub(Tensor *A, Tensor *B, float v){
    Tensor::add(A, B, -1.0f * v);
}


void Tensor::tan_(){
    Tensor::tan(this, this);
}


Tensor* Tensor::tan(){
    Tensor *t = this->clone();
    t->tan_();
    return t;
}


void Tensor::tan(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(tan);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_tan(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_tan(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_tan(A, B);
      }
#endif

    PROFILING_FOOTER(tan);
}


void Tensor::tanh_(){
    Tensor::tanh(this, this);
}


Tensor* Tensor::tanh(){
    Tensor *t = this->clone();
    t->tanh_();
    return t;
}


void Tensor::tanh(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(tanh);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_tanh(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_tanh(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_tanh(A, B);
      }
#endif

    PROFILING_FOOTER(tanh);
}


void Tensor::trunc_(){
    Tensor::trunc(this, this);
}


Tensor* Tensor::trunc(){
    Tensor *t = this->clone();
    t->trunc_();
    return t;
}


void Tensor::trunc(Tensor *A, Tensor *B){

    PROFILING_HEADER_EXTERN(trunc);
    
    if (A->isCPU() && B->isCPU()) {
        cpu_trunc(A, B);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_trunc(A, B);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA() && B->isFPGA())
      {
        fpga_trunc(A, B);
      }
#endif

    PROFILING_FOOTER(trunc);
}


// Math operations (binary) ************************
Tensor* Tensor::add(Tensor *A, Tensor *B){
    Tensor* C = Tensor::empty(A->getShape(), A->device);
    Tensor::add(A, B, C);
    return C;
}

void Tensor::add(Tensor *A, Tensor *B, Tensor *C) {
    Tensor::add(1.0, A, 1.0, B, C, 0);
}


Tensor* Tensor::div(Tensor *A, Tensor *B){
    Tensor* C = Tensor::empty(A->getShape(), A->device);
    Tensor::div(A, B, C);
    return C;
}

void Tensor::div(Tensor *A, Tensor *B, Tensor *C) {
    Tensor::el_div(A, B, C, 0);
}


Tensor* Tensor::mult(Tensor *A, Tensor *B){
    Tensor* C = Tensor::empty(A->getShape(), A->device);
    Tensor::mult(A, B, C);
    return C;
}


void Tensor::mult(Tensor *A, Tensor *B, Tensor *C) {
    Tensor::el_mult(A, B, C, 0);
}


Tensor* Tensor::interpolate(float factor1, Tensor *A, float factor2, Tensor *B){
    Tensor* C = Tensor::empty(A->getShape(), A->device);
    Tensor::interpolate(factor1, A, factor2, B, C);
    return C;
}

void Tensor::interpolate(float factor1, Tensor *A, float factor2, Tensor *B, Tensor *C){
    Tensor::add(factor1, A, factor2, B, C, 1);
}


Tensor* Tensor::sub(Tensor *A, Tensor *B){
    Tensor* C = Tensor::empty(A->getShape(), A->device);
    Tensor::sub(A, B, C);
    return C;
}

void Tensor::sub(Tensor *A, Tensor *B, Tensor *C) {
    Tensor::add(1.0, A, -1.0, B, C, 0);
}


// ***** Overload operators *****************************
// Tensor and Tensor (Element wise)  ********************
Tensor& operator+ (Tensor &A, Tensor &B) {
    Tensor* t = Tensor::add(&A, &B);
    return (*t);
}


Tensor& operator- (Tensor &A, Tensor &B) {
    Tensor* t = Tensor::sub(&A, &B);
    return (*t);
}


Tensor& operator* (Tensor &A, Tensor &B) {
    Tensor* t = Tensor::mult(&A, &B);
    return (*t);
}


Tensor& operator/ (Tensor &A, Tensor &B) {
    Tensor* t = Tensor::div(&A, &B);
    return (*t);
}


// Tensor op= Tensor (Element wise)  ********************
void operator+= (Tensor &A, Tensor &B) {
    Tensor::add(1.0f, &A, 1.0f, &B, &A, 0);
}


void operator-= (Tensor &A, Tensor &B) {
    Tensor::add(1.0f, &A, -1.0f, &B, &A, 0);
}


void operator*= (Tensor &A, Tensor &B) {
    Tensor::el_mult(&A, &B, &A, 0);
}


void operator/= (Tensor &A, Tensor &B) {
    Tensor::el_div(&A, &B, &A, 0);
}

// Tensor op= Scalar  ********************
void operator+= (Tensor &A, float v) {
    A.add_(v);
}


void operator-= (Tensor &A, float v) {
    A.sub_(v);
}


void operator*= (Tensor &A, float v) {
    A.mult_(v);
}


void operator/= (Tensor &A, float v) {
    A.div_(v);
}


// Tensor and scalar *******************
Tensor& operator+ (Tensor &A, float v) {
    Tensor* t = A.clone();
    t->add_(v);
    return (*t);
}


Tensor& operator- (Tensor &A, float v) {
    Tensor* t = A.clone();
    t->add_(-v);
    return (*t);
}


Tensor& operator* (Tensor &A, float v) {
    Tensor* t = A.clone();
    t->mult_(v);
    return (*t);
}


Tensor& operator/ (Tensor &A, float v) {
    Tensor* t = A.clone();
    t->div_(v);
    return (*t);
}


// Scalar and Tensor *******************
Tensor& operator+ (float v, Tensor &A) {
    return A + v;
}


Tensor& operator- (float v, Tensor &A) {
    Tensor* t = A.clone();
    t->neg_();
    t->add_(v);
    return (*t);
}


Tensor& operator* (float v, Tensor &A) {
    return A * v;
}


Tensor& operator/ (float v, Tensor &A) {
    Tensor* t = A.clone();
    t->inv_(v);
    return (*t);
}





void Tensor::add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {
    ///////////////////////////////////////
    //// sum C=(sca*A)+(scb*B)
    //// or C+=(sca*A)+(scb*B) if incC is 1
    //// Dimensions and types must be compatible
    ///////////////////////////////////////
    int aux = 0;

    PROFILING_HEADER_EXTERN(add);

    if ((A->device != B->device) || (A->device != C->device)) msg("Tensors in different devices", "Tensor::add_");
    if ((!sameShape(A, B)) || (!sameShape(A, C))) {
        A->info();
        B->info();
        C->info();
        msg("Incompatible dims", "Tensor::add");
    }


    if (A->isCPU()) {
        cpu_add(scA, A, scB, B, C, incC);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_add(scA, A, scB, B, C, incC);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA())
      {
        fpga_add(scA, A, scB, B, C, incC);
      }
#endif



    PROFILING_FOOTER(add);
}


void Tensor::inc(Tensor *A, Tensor *B) {
    // TODO: Review against add

    PROFILING_HEADER_EXTERN(inc);

    if (!Tensor::sameShape(A, B))
        msg("Tensors with different shape", "Tensor::inc");


    if ((A->isCPU()) && (B->isCPU())) {
        cpu_inc(A, B);
    }
#ifdef cGPU
    else if ((A->isGPU())&&(B->isGPU())) {
        Tensor::add(1,A,1,B,B,0);
    }
    else if (((A->isCPU())&&(B->isGPU()))||((A->isGPU())&&(B->isCPU())))
    {
        Tensor *n=new Tensor(B->getShape(),B->device);
        Tensor::copy(A,n);
        Tensor::add(1,n,1,B,B,0);
        delete n;
    }
#endif
    else if ((A->isFPGA())&&(B->isFPGA())) {
        Tensor::add(1,A,1,B,B,0);
    }
    else if (((A->isCPU())&&(B->isFPGA()))||((A->isFPGA())&&(B->isCPU())))
    {   
        Tensor *n=new Tensor(B->getShape(),B->device);
        Tensor::copy(A,n);
        Tensor::add(1,n,1,B,B,0);
        delete n;
    }

    PROFILING_FOOTER(inc);
}

void Tensor::el_div(Tensor *A, Tensor *B, Tensor *C, int incC) {
    ///////////////////////////////////////
    //// Element Div C=A./B
    //// incC 1 means C+=A./B (increment over C)
    //// Dimensions must be compatible
    ///////////////////////////////////////

    if ((A->device != B->device) || (A->device != C->device)) msg("Tensors in different devices", "Tensor::el_div");
    if ((!sameShape(A, B)) || (!sameShape(A, C))) msg("Incompatible dims", "Tensor::el_div");

    PROFILING_HEADER_EXTERN(el_div);


    if (A->isCPU()) {
        cpu_el_div(A, B, C, incC);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_el_div(A,B,C,incC);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA())
      {
        fpga_el_div(A,B,C,incC);
      }
#endif


    PROFILING_FOOTER(el_div);
}


void Tensor::mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC) {
    ///////////////////////////////////////
    //// MULT2D C=A*B
    //// tA means transpose A {0,1}
    //// tB means transpose B {0,1}
    //// tC 1 means C+=A*B (increment over C)
    //// Dimensions and types must be compatible
    //// Only for 2D Tensors
    ///////////////////////////////////////
   
    PROFILING_HEADER_EXTERN(mult2D);

    if ((A->device != B->device) || (A->device != C->device)) {A->info();B->info();C->info();msg("Tensors in different devices", "Tensor::mult2D");}
    if ((A->ndim != 2) || (B->ndim != 2) || (C->ndim != 2)) msg("Only 2D tensors", "Tensor::mult2D");
    if (!tA) {
        if (!tB) {
            if ((A->shape[1] != B->shape[0]) || (A->shape[0] != C->shape[0]) || (B->shape[1] != C->shape[1]))
                msg("Incompatible dims", "Tensor::mult2D");
        } else if ((A->shape[1] != B->shape[1]) || (A->shape[0] != C->shape[0]) || (B->shape[0] != C->shape[1]))
            msg("Incompatible dims", "Tensor::mult2D");
    } else {
        if (!tB) {
            if ((A->shape[0] != B->shape[0]) || (A->shape[1] != C->shape[0]) || (B->shape[1] != C->shape[1]))
                msg("Incompatible dims", "Tensor::mult2D");
        } else if ((A->shape[0] != B->shape[1]) || (A->shape[1] != C->shape[0]) || (B->shape[0] != C->shape[1]))
            msg("Incompatible dims", "Tensor::mult2D");
    }




    if (A->isCPU()) {
        cpu_mult2D(A, tA, B, tB, C, incC);
    }

#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_mult2D(A,tA,B,tB,C,incC);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA())
      {
        fpga_mult2D(A,tA,B,tB,C,incC);
      }
#endif


    PROFILING_FOOTER(mult2D);
}


void Tensor::el_mult(Tensor *A, Tensor *B, Tensor *C, int incC) {
    ///////////////////////////////////////
    //// Element Mult C=A.*B
    //// incC 1 means C+=A.*B (increment over C)
    //// Dimensions must be compatible
    ///////////////////////////////////////

    PROFILING_HEADER_EXTERN(el_mult);


    if ((A->device != B->device) || (A->device != C->device)) msg("Tensors in different devices", "Tensor::el_mult");
    if ((!sameShape(A, B)) || (!sameShape(A, C))) {
        A->info();
        B->info();
        C->info();
        msg("Incompatible dims", "Tensor::el_mult");
    }

    if (A->isCPU()) {
        cpu_el_mult(A, B, C, incC);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
         gpu_el_mult(A,B,C,incC);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA())
      {
         fpga_el_mult(A,B,C,incC);
      }
#endif


    PROFILING_FOOTER(el_mult);    
}


void Tensor::sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C) {
    ///////////////////////////////////////
    //// sum2D_rowise C=A.rowise+B
    //// Dimensions and types must be compatible
    //// A is 2D Tensor
    //// B is 1D Tensor
    ///////////////////////////////////////
    if ((A->device != B->device) || (A->device != C->device))
        msg("Tensors in different devices", "Tensor::sum2D_rowwise");
    if ((A->ndim != 2) || (B->ndim != 1) || (C->ndim != 2)) msg("sum2D_rowwise dims");
    if ((!sameShape(A, C)) || (A->shape[1] != B->shape[0])) msg("Incompatible dims", "Tensor::sum2D_rowwise");

    PROFILING_HEADER_EXTERN(sum2D_rowwise);


    if (A->isCPU()) {
        cpu_sum2D_rowwise(A, B, C);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_sum2D_rowwise(A,B,C);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA())
      {
        fpga_sum2D_rowwise(A,B,C);
      }
#endif


    PROFILING_FOOTER(sum2D_rowwise);
}


void Tensor::reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB) {
    ///////////////////////////////////////
    //// reduce_sum2D B=reduce_sum2D(A)
    //// Dimensions and types must be compatible
    //// A is 2D Tensor
    //// B is 1D Tensor
    //// axis is the dimension to be sumed
    ///////////////////////////////////////
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::reduce_sum2D");
    if ((A->ndim - 1) != B->ndim) msg("Incorrect dims", "Tensor::reduce_sum2D");
    if ((A->shape[1 - axis] != B->shape[0])) msg("Incompatible dims", "Tensor::reduce_sum2D");

    PROFILING_HEADER_EXTERN(reduce_sum2D);


    if (A->isCPU()) {
        cpu_reduce_sum2D(A, B, axis, incB);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_reduce_sum2D(A,B,axis,incB);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA())
      {
        fpga_reduce_sum2D(A,B,axis,incB);
      }
#endif


    PROFILING_FOOTER(reduce_sum2D);    
}

void Tensor::sum2D_colwise(Tensor *A, Tensor *B, Tensor *C) {
    ///////////////////////////////////////
    //// sum2D_colwise C=A.colwise+B
    //// Dimensions and types must be compatible
    //// A is 2D Tensor
    //// B is 1D Tensor
    ///////////////////////////////////////
    if ((A->device != B->device) || (A->device != C->device))
        msg("Tensors in different devices", "Tensor::sum2D_colwise");
    if ((A->ndim != 2) || (B->ndim != 1) || (C->ndim != 2)) msg("sum2D_colwise dims");
    if ((!sameShape(A, C)) || (A->shape[0] != B->shape[0])) msg("Incompatible dims", "Tensor::sum2D_colwise");

    PROFILING_HEADER_EXTERN(sum2D_colwise);


    if (A->isCPU()) {
        cpu_sum2D_colwise(A, B, C);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_sum2D_colwise(A,B,C);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA())
      {
        fpga_sum2D_colwise(A,B,C);
      }
#endif


    PROFILING_FOOTER(sum2D_colwise);    
}
