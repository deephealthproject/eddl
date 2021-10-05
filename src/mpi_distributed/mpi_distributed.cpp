/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   mpi_distributed.h
 * Author: plopez
 *
 * Created on 27 de julio de 2021, 11:18
 */

#include "eddl/mpi_distributed/mpi_distributed.h"

#ifdef cCUDA
#include "cuda.h"
#endif

// Global variables
int use_mpi = 0;
int mpi_avg = 1;
int avg_method = 0;
int x_avg = 0;
int batch_is_global=1; 
// 1: Global batch=batch; Local batch=batch/n_procs 
// 0: Local batch=batch; Global_batch=batch*n_procs

#ifdef cNCCL
// NCCL
ncclUniqueId nccl_id;
ncclComm_t nccl_comm;
cudaStream_t cuda_stream;
#endif

int get_id_distributed() {    
    int id=0;
#ifdef cMPI
    //  Get the individual process ID.
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
#endif
    return id;
}

int get_n_procs_distributed() { 
    int n_procs=1;
#ifdef cMPI
   //  Get the number of processes.
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
#endif
    return n_procs;
}

int init_distributed(int *argc, char ***argv) {
    int n_procs;
    int id;
    
#ifndef cMPI
    msg("MPI library is not linked", "init_distributed");
#endif  
    
    id=0;
#ifdef cMPI
    MPI_Init(argc, argv);
    
    use_mpi = 1;
    
   
    n_procs=get_n_procs_distributed();
    id=get_id_distributed();
 
    fprintf(stderr, "[DISTR] setting default method\n");
    set_method_distributed(FIXED, AVG_DEFAULT, 0);
    
    // Initalize a different seed per proc
    srand(id);
#endif

#ifdef cNCCL
    //NCCL
    //get NCCL unique ID at rank 0 and broadcast it to all others
    if (id == 0) ncclGetUniqueId(&nccl_id);
    MPICHECK(MPI_Bcast(&nccl_id, sizeof (nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
    //picking a GPU based on localRank, allocate device buffers
    CUDACHECK(cudaStreamCreate(&cuda_stream));
    //initializing NCCL
    NCCLCHECK(ncclCommInitRank(&nccl_comm, n_procs, nccl_id, id));
    if (id == 0)
        fprintf(stdout, "[DISTR] NCCL initialized %d procs\n", n_procs);
#endif

    return id;
}

void set_method_distributed (int method, int batch_avg, int epoch_avg) {    
    int n_procs;
    int id;
    
#ifndef cMPI
    msg("MPI library is not linked", "set_method_distributed");
#endif  

    avg_method= method;
    mpi_avg = batch_avg;
    x_avg= epoch_avg;
    
    n_procs=get_n_procs_distributed();
    id=get_id_distributed();

    if (id == 0)
        if (avg_method == FIXED) {
            fprintf(stderr, "[DISTR] %d procs. method %s batch_sync %d \n", n_procs, "FIXED", mpi_avg);
        } else if (avg_method == AVG_INC) {
            fprintf(stderr, "[DISTR] %d procs. method %s batch_sync %d changing every %d epochs\n", n_procs, "AVG_INC", mpi_avg, x_avg);
        } else if (avg_method == SAWTOOTH) {
            fprintf(stderr, "[DISTR] %d procs. method %s batch_sync %d changing every %d epochs\n", n_procs, "SAWTOOTH", mpi_avg, x_avg);
        } else if (avg_method == NEG_SAWTOOTH) {
            fprintf(stderr, "[DISTR] %d procs. method %s batch_sync %d changing every %d epochs\n", n_procs, "NEG SAWTOOTH", mpi_avg, x_avg);
        } else if (avg_method == AUTO_TIME) {
            fprintf(stderr, "[DISTR] %d procs. method %s batch_sync %d changing every %d epochs\n", n_procs, "AUTO TIME", mpi_avg, x_avg);
        }else {
            fprintf(stderr, "[DISTR] Error sync method %d not implemented\n", avg_method);
            exit(EXIT_FAILURE);
        } 
}

void end_distributed() {
    int id;
#ifndef cMPI
    msg("MPI library is not linked", "end_distributed");
#endif    

#ifdef cMPI
    if (use_mpi) {
        //  Get the individual process ID.
        MPI_Comm_rank(MPI_COMM_WORLD, &id);
#endif
  
#ifdef cNCCL
        //finalizing NCCL
        ncclCommDestroy(nccl_comm);
        if (id == 0)
            fprintf(stdout, "[DISTR] NCCL End\n");
#endif

#ifdef cMPI
        if (id == 0)
            fprintf(stdout, "[DISTR] End\n");
        MPI_Finalize();
    }
#endif
}

void fn_mpi_AllReduce(float* myptr, int count) {
#ifndef cMPI
    msg("invalid call to MPI_Allreduce. MPI library is not linked", "fn_mpi_AllReduce");
#endif
    
#ifdef cMPI
    if (count > 0) {
        MPICHECK(MPI_Allreduce(MPI_IN_PLACE, myptr, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
    }
#endif
}

void fn_nccl_AllReduce(float* myptr, int count) {
#ifdef cNCCL
    if (count > 0) {
        NCCLCHECK(ncclAllReduce((const void*) myptr, (void*) myptr, count, ncclFloat, ncclSum, nccl_comm, cuda_stream));
        //completing NCCL operation by synchronizing on the CUDA stream
        CUDACHECK(cudaStreamSynchronize(cuda_stream));
    }
#endif
}

void AllReduce_distributed(float* myptr, int count) {
#ifdef cNCCL
    fn_nccl_AllReduce(myptr, count); 
#else
    fn_mpi_AllReduce(myptr, count);
#endif
}

int get_local_GPU_distributed(int id, int nGPUs) {
    int nDevices=1;
//#ifdef cCUDA
//    cudaGetDeviceCount(&nDevices);
//#endif
//    return id % nDevices;
    return id % nGPUs;
}

int is_mpi_distributed() {
    return use_mpi;
}

int get_params_distributed(int* method, int* avg, int* avg_chg, int* batch_global) {

    *avg = mpi_avg;
    *method = avg_method;
    *avg_chg = x_avg;
    *batch_global = batch_is_global;

    return use_mpi;
}