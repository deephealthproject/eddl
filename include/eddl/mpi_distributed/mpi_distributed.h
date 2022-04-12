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
#include "omp.h"

#ifdef cGPU
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

#ifdef cMPI
#include <mpi.h>
#include <mpi-ext.h> /* Needed for CUDA-aware check */
#endif

#ifdef cNCCL
#include <nccl.h>
#endif

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define mpi_id0(...)   \
    if (id==0) {\
        __VA_ARGS__; \
    }


#define AVG_DEFAULT 16

#define FIXED 0
#define AVG_INC 1
#define SAWTOOTH 2
#define NEG_SAWTOOTH 3
#define AUTO_TIME 4


#define DIV_BATCH 0
#define MUL_BATCH 1

#define NO_DISTR_DS 0
#define DISTR_DS 1



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
 *  @brief Get name of node running the process
 *
 *  @param node_name node name
 */
void get_nodename_distributed(char* node_name);

/**
 *  @brief Get boolean vector for GPU 
 *  GPUs are assigned to processes following a module operation id % nr_gpus
 * 
 *  @return boolean vector with assinged GPU
 */
vector<int> get_gpu_vec_distributed();

/**
 *  @brief Initializes distributed training
 *
 *  @param comm  NCCL MPI
 *  @return id MPI rank of process
 */
int init_distributed(string comm);

/**
 *  @brief Initializes distributed training
 *  NCCL is used
 *  @return id MPI rank of process
 */
int init_distributed();

//int init_MPI();
//void init_NCCL(int nr_gpus);

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

/**
 *  @brief Get current batches_avg value
 *
 *  @return batches_avg
 */
int get_current_batch_avg_distributed ();


/**
 *  @brief Get nr of GPUs per node
 *
 *  @return nr of GPUs
 */
int get_available_GPUs_distributed();


/**
 *  @brief Get nr of CPUs per node
 *
 *  @return nr of CPUs
 */
int get_available_CPUs_distributed();

void set_OMP_threads_to_procs_distributed();

/**
 *  @brief Set per-process batch size
 *
 *  @param[in/out] global_batch: global batch size
 *  @param[in/out] local_batch: per-process batch size
 *  @param[in] batch: programmer batch size
 *  @param[in] method: DIV_BATCH (local_batch=batch/n) or MUL_BATCH (local_batch=batch)
 */
void set_batch_distributed (int* global_batch, int* local_batch, int batch, int method);

/**
 *  @brief Set number of batches per proc 
 *
 *  @param[in] ds_size: datset size
 *  @param[in] local_batch: per-process batch size
 *  @param[in] method: DISTR_DS or NO_DISTR_DS
 *  @return nr of batches per process
 */
int set_NBPP_distributed (int ds_size, int local_batch, bool method);


/**
 *  @brief All reduce loss & accuracy of last layer 
 *
 *  @param[in] net: Ptr to net
 */
void avg_metrics_distributed(Net* net);


/**
 *  @brief All reduce float variable
 *
 *  @param[in] pvar: Ptr to variable
 */
void avg_float_distributed(float* pvar);


/**
 *  @brief Call to MPI_Barrier
 *  
 */
void barrier_distributed();

/**
 *  @brief Broadcast net parameters
 *
 *  @param[in] net: Ptr to net
 */
void bcast_weights_distributed(Net * net);

/**
 *  @brief Averages weights and bias of mpi processes
 *
 *  @param curr_batch_id    Batch nr (from 1)
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

/**
 *  @brief Update batches_avg according to max comm overhead
 *
 *  @param secs_train:  time in secs to train (only computation)
 *  @param secs_comm:   time in secs to communicate
 *  @param overhead:    max comm overhead (0 to 1)
 *  @return    (void)
 */
void set_batch_avg_overhead_distributed(double secs_train, double secs_comm, float overhead, int max_ba);

// For Debugging purposes
void gpu_layer_print (Net* net, int layer);

bool early_stopping_on_loss_var(Net* net, int index, float delta, int patience, int epoch);

bool early_stopping_on_metric_var(Net* net, int index, float delta, int patience, int epoch);

bool early_stopping_on_metric(Net* net, int index, float goal, int patience, int epoch);

void GPU_quantize_network_distributed(Net* net, int nbits_int, int nbits_frac);

void CPU_quantize_network_distributed(Net* net, int nbits_int, int nbits_frac);
