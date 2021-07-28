/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */



#ifndef MPI_DISTRIBUTED_H
#define MPI_DISTRIBUTED_H
#endif /* MPI_DISTRIBUTED_H */


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#ifdef cMPI
#include <mpi.h>
#endif

#ifdef cNCCL
#include <nccl.h>
#endif

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

/**
 *  @brief Initializes distributed training
 *
 *  @param argc,argv Command line arguments
 *  @param avg Nr of batches between parameters synchronization
 *  @param method Method to sinchronize (0: constant)
 *  @return id MPI rank of process 
 */
int init_distributed(int *argc, char ***argv, int avg, int method);
/**
 *  @brief Finalizes distributed training
 *
 */
void end_distributed();
    
/**
 *  @brief Performs AllReduction of buffer using MPI
 * 
 *  @param myptr pointer to buffer
 *  @param count buffer size in floats
 */
void fn_mpi_AllReduce(float* myptr, int count);
/**
 *  @brief Performs AllReduction of buffer using NCCL
 * 
 *  @param myptr pointer to buffer
 *  @param count buffer size in floats
 */
void fn_nccl_AllReduce(float* myptr, int count);





