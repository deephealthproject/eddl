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

int use_mpi = 0;
int mpi_avg = 1;
int avg_method = 0;
int x_avg = 0;

#ifdef cNCCL
// NCCL
ncclUniqueId nccl_id;
ncclComm_t nccl_comm;
cudaStream_t cuda_stream;
#endif

int fn_get_id() {
    int id=0;
#ifdef cMPI
    //  Get the individual process ID.
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
#endif
    return id;
}

int fn_get_n_procs() {
    int n_procs=0;
#ifdef cMPI
   //  Get the number of processes.
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
#endif
    return n_procs;
}

int init_distributed(int *argc, char ***argv) {
    int n_procs;
    int id;

    id=0;
#ifdef cMPI
    MPI_Init(argc, argv);
    
    use_mpi = 1;
    
    mpi_avg = AVG_DEFAULT;
    avg_method = FIXED;
    
    //  Get the number of processes.
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    //  Get the individual process ID.
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    srand(id);
    
    if (id == 0)
        if (avg_method == 0) {
            fprintf(stderr, "[DISTR(default)]  %d procs. method %s batch_sync %d \n", n_procs, "FIXED", mpi_avg);
        } else if (avg_method == 1) {
            fprintf(stderr, "[DISTR(default)] DEFAULT %d procs. method %s batch_sync %d grows every %d epochs\n", n_procs, "AVG_INC", mpi_avg, x_avg);
        } else if (avg_method == 2) {
            fprintf(stderr, "[DISTR(default)] %d procs. method %s batch_sync %d grows every %d epochs\n", n_procs, "SAWTOOTH", mpi_avg, x_avg);
        } else if (avg_method == 3) {
            fprintf(stderr, "[DISTR(default)] %d procs. method %s batch_sync %d reduces every %d epochs\n", n_procs, "NEG SAWTOOTH", mpi_avg, x_avg);
        } else if (avg_method == 4) {
            fprintf(stderr, "[DISTR(default)] %d procs. method %s batch_sync %d grows every %d epochs\n", n_procs, "AUTO TIME", mpi_avg, x_avg);
        }else {
            fprintf(stderr, "[DISTR] Error sync method %d not implemented\n", avg_method);
            exit(EXIT_FAILURE);
        } 
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

void set_method_distributed (int method, int batch_avg, int epoch_avg) 
{
 
    avg_method= method;
    mpi_avg = batch_avg;
    x_avg= epoch_avg;

    if (fn_get_id() == 0)
        if (avg_method == 0) {
            fprintf(stderr, "[DISTR] %d procs. method %s batch_sync %d \n", fn_get_n_procs(), "FIXED", mpi_avg);
        } else if (avg_method == 1) {
            fprintf(stderr, "[DISTR] %d procs. method %s batch_sync %d changing every %d epochs\n", fn_get_n_procs(), "AVG_INC", mpi_avg, x_avg);
        } else if (avg_method == 2) {
            fprintf(stderr, "[DISTR] %d procs. method %s batch_sync %d changing every %d epochs\n", fn_get_n_procs(), "SAWTOOTH", mpi_avg, x_avg);
        } else if (avg_method == 3) {
            fprintf(stderr, "[DISTR] %d procs. method %s batch_sync %d changing every %d epochs\n", fn_get_n_procs(), "NEG SAWTOOTH", mpi_avg, x_avg);
        } else if (avg_method == 4) {
            fprintf(stderr, "[DISTR] %d procs. method %s batch_sync %d changing every %d epochs\n", fn_get_n_procs(), "AUTO TIME", mpi_avg, x_avg);
        }else {
            fprintf(stderr, "[DISTR] Error sync method %d not implemented\n", avg_method);
            exit(EXIT_FAILURE);
        } 
}

void end_distributed() {
    int id;

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
#endif

    }
}

void fn_mpi_AllReduce(float* myptr, int count) {
#ifdef cMPI
    if (count > 0) {
        MPICHECK(MPI_Allreduce(MPI_IN_PLACE, myptr, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
    }
#endif
}

void fn_nccl_AllReduce(float* myptr, int count) {
#ifdef cNCCL
    if (count > 0) {
        CUDACHECK(cudaSetDevice(0));
        NCCLCHECK(ncclAllReduce((const void*) myptr, (void*) myptr, count, ncclFloat, ncclSum, nccl_comm, cuda_stream));
        //completing NCCL operation by synchronizing on the CUDA stream
        //CUDACHECK(cudaSetDevice(0));
        CUDACHECK(cudaStreamSynchronize(cuda_stream));
    }
#endif
}


