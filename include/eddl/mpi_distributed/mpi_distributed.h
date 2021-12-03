/*
* MPI support for EDDL Library - European Distributed Deep Learning Library.
* Version: 
* copyright (c) 2021, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: July 2021
* Author: GAP Research Group (UPV), contact: plopez@disca.upv.es
* All rights reserved
*/



#ifndef MPI_DISTRIBUTED_H
#define MPI_DISTRIBUTED_H
#endif /* MPI_DISTRIBUTED_H */


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include "eddl/apis/eddl.h"
#include "eddl/net/net.h"
#include "eddl/layers/core/layer_core.h"


#ifdef cMPI
#include <mpi.h>
#endif

#ifdef cNCCL
#include <nccl.h>
#endif

#define AVG_DEFAULT 16

#define FIXED 0
#define AVG_INC 1
#define SAWTOOTH 2
#define NEG_SAWTOOTH 3
#define AUTO_TIME 4

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

#define GPU_1_distributed \
    switch (id % 1) { \
        case 0: gpus={1}; \
            break; \
        }  

#define GPU_2_distributed \
    switch (id % 2) { \
        case 0: gpus={1, 0}; \
            break; \
        case 1: gpus={0, 1}; \
            break; \
        }          

#define GPU_4_distributed \
    switch (id % 4) { \
        case 0: gpus={1, 0, 0, 0}; \
            break; \
        case 1: gpus={0, 1, 0, 0}; \
            break; \
        case 2: gpus={0, 0, 1, 0}; \
            break; \
        case 3: gpus={0, 0, 0, 1}; \
            break; \
        }          

#define GPU_8_distributed \
    switch (id % 8) { \
        case 0: gpus={1, 0, 0, 0, 0, 0, 0, 0}; \
            break; \
        case 1: gpus={0, 1, 0, 0, 0, 0, 0, 0}; \
            break; \
        case 2: gpus={0, 0, 1, 0, 0, 0, 0, 0}; \
            break; \
        case 3: gpus={0, 0, 0, 1, 0, 0, 0, 0}; \
            break; \
        case 4: gpus={0, 0, 0, 0, 1, 0, 0, 0}; \
            break; \
        case 5: gpus={0, 0, 0, 0, 0, 1, 0, 0}; \
            break; \
        case 6: gpus={0, 0, 0, 0, 0, 0, 1, 0}; \
            break; \
        case 7: gpus={0, 0, 0, 0, 0, 0, 0, 1}; \
            break; \
        }         


/**
 *  @brief Get MPI id of process
 *
 *  @return id MPI rank of process
 */
int get_id_distributed();

/**
 *  @brief Get nr of MPI running processes
 *
 *  @return nr of MPI processes
 */
int get_n_procs_distributed();

/**
 *  @brief Initializes distributed training
 *
 *  @param argc,argv Command line arguments
 *  @return id MPI rank of process
 */
int init_distributed(int *argc, char ***argv);
int init_distributed();

int init_MPI();
int init_NCCL(int nr_gpus);


/**
 *  @brief Sets distributed training paramas
 *
 *  @param method Method to sinchronize
 *  @param batch_avg Nr of batches between parameters synchronization
 *  @param epoch_avg Nr of epochs between changes in batch_avg
 */
void set_method_distributed(int method, int batch_avg, int epoch_avg);

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

/**
 *  @brief Performs AllReduction of buffer using NCCL if available, MPI otherwise
 *
 *  @param myptr pointer to buffer
 *  @param count buffer size in floats
 */
void AllReduce_distributed(float* myptr, int count);

void AllReduce_streams_distributed(float* myptr, int count, int layer);


/**
 *  @brief Checks if running in mpi_distributed mode
 *
 *  @return true if running in mpi_distributed mode
 */
int is_mpi_distributed();

/**
 *  @brief Get MPI distributed params
 *
 *  @param[out] avg_method: Method to compute average of params
 *  @param[out] mpi_avg:    Elapsed nr of batches among averages
 *  @param[out] x_avg:      Elapsed nr of batches to change mpi_avg
 *  @param[out] batch_is_global: True if batch size is interpreted as global batch
 * 
 *  @return true if running in mpi_distributed mode
 */
int get_params_distributed(int* method, int* avg, int* avg_chg);


int get_current_batch_avg_distributed ();


int get_available_GPUs_distributed();

/**
 *  @brief Broadcast net parameters
 *
 *  @param[in] net: Ptr to net
 */
void Bcast_params_distributed(Net * net);

/**
 *  @brief Average weights of mpi processes
 *
 *  @param curr_batch_id    Batch nr (from 0)
 *  @param batches_per_proc #batches per mpi process
 *  @return    (void)
 */
void avg_weights_distributed (Net* net, int curr_batch_id, int batches_per_proc);

/**
 *  @brief Update batches_avg according to the selected method
 *
 *  @param epoch_id:            epoch index (from 0)
 *  @param secs:                elapsed time in current epoch
 *  @param batches_per_proc:    nr of batches in every MPI process
 *  @return    (void)
 */
void update_batch_avg_distributed(int epoch_id, double secs, int batches_per_proc) ;

void gpu_layer_print (Net* net, int layer);

