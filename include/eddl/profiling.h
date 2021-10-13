/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Polit�cnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef _PROFILING
#define _PROFILING

#include "eddl/system_info.h"

#ifdef EDDL_WINDOWS

#include <chrono>
#include <windows.h>
#include <winsock.h>

int gettimeofday(struct timeval* tp, struct timezone* tzp);

#else
#include <sys/time.h>
#endif

//#define PROFILING_HEADER(fn) \
//    struct timeval _##_t1; \
//    gettimeofday(&prof_t1, NULL);

#define PROFILING_ENABLE(fn) \
    unsigned long long prof_##fn##_time; \
    unsigned long long prof_##fn##_calls;

#define PROFILING_ENABLE_EXTERN(fn) \
    extern unsigned long long prof_##fn##_time; \
    extern unsigned long long prof_##fn##_calls; \

#define PROFILING_HEADER(fn) \
    struct timeval _##fn##_t1; \
    gettimeofday(&_##fn##_t1, NULL);

#define PROFILING_HEADER_EXTERN(fn) \
    extern unsigned long long prof_##fn##_time; \
    extern unsigned long long prof_##fn##_calls; \
    struct timeval _##fn##_t1; \
    gettimeofday(&_##fn##_t1, NULL);

#define PROFILING_FOOTER(fn) \
    struct timeval _##fn##_t2; \
    gettimeofday(&_##fn##_t2, NULL); \
    prof_##fn##_time += ((_##fn##_t2.tv_sec - _##fn##_t1.tv_sec) * 1000000) + (_##fn##_t2.tv_usec - _##fn##_t1.tv_usec); \
    prof_##fn##_calls += 1;

#define PROFILING_PRINTF(fn) \
    if (prof_##fn##_calls > 0) printf("  %-50s: %8lld calls, %12lld us , %10.4f us/call\n", #fn, \
                    prof_##fn##_calls, prof_##fn##_time, \
                    (float) prof_##fn##_time / (float) prof_##fn##_calls);

#define PROFILING_PRINTF2(fn, acc) \
    if (prof_##fn##_calls > 0) printf("  %-50s: %8lld calls, %12lld us (%6.2f), %10.4f us/call\n", #fn, \
                    prof_##fn##_calls, prof_##fn##_time, \
            100.0 * prof_##fn##_time / acc, (float) prof_##fn##_time / (float) prof_##fn##_calls);
#endif

#define PROFILING_RESET(fn) \
  prof_##fn##_calls = 0; \
  prof_##fn##_time = 0;


//CxHxW
//
//HxWxC
//
//GxHxWxC (C=4)   Reshape + Permute
//
//32xHxW -> Reshape -> 8x4xHxW -> Permute(0, 2, 3, 1) -> 8xHxWx4   // hay capas y funciones
