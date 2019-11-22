/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>

#include "gpu_kernels.h"


/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

  /* we have to initialize the state */
  curand_init((unsigned int)clock64(), /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void random_uniform(curandState_t* states, float* numbers) {
  /* curand works like rand - except that it takes a state as a parameter */
  numbers[blockIdx.x] = curand_uniform(&states[blockIdx.x]);
}
