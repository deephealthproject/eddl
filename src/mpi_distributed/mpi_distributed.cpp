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
#ifdef cNCCL
// NCCL
ncclUniqueId nccl_id;
ncclComm_t nccl_comm;
cudaStream_t cuda_stream;
#endif

int init_distributed(int *argc, char ***argv, int avg, int method) {
    int n_procs;
    int id;

    id=0;
#ifdef cMPI
    MPI_Init(argc, argv);
    use_mpi = 1;
    mpi_avg = avg;
    avg_method = method;
    //  Get the number of processes.
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    //  Get the individual process ID.
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    srand(id);
    
    if (id == 0)
        if (method == 0) {
            fprintf(stderr, "[DISTR] %d procs, method %s sync every %d batches \n", n_procs, "fixed ", mpi_avg);
        } else if (method == 1) {
            fprintf(stderr, "[DISTR] %d procs, method %s  initial sync every %d batches \n", n_procs, "dynamic ", mpi_avg);
        } else if (method == 2) {
            fprintf(stderr, "[DISTR] %d procs, method %s  initial sync every %d batches \n", n_procs, "sawtooth ", mpi_avg);
        } else if (method == 3) {
            fprintf(stderr, "[DISTR] %d procs, method %s  initial sync every %d batches \n", n_procs, "negative sawtooth ", mpi_avg);
        } else if (method == 4) {
            fprintf(stderr, "[DISTR] %d procs, method %s  initial sync every %d batches \n", n_procs, "adaptive ", mpi_avg);
        } else {
            fprintf(stderr, "[DISTR] Error sync method %d not implemented\n", method);
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


