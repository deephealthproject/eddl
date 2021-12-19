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
 *  @return node name
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
 *  @brief Broadcast net parameters
 *
 *  @param[in] net: Ptr to net
 */
void bcast_weights_distributed(Net * net);

/**
 *  @brief Averages weights and bias of mpi processes
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

// For Debugging purposes
void gpu_layer_print (Net* net, int layer);

