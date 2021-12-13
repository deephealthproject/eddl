/*
 * MPI support for EDDL Library - European Distributed Deep Learning Library.
 * Version: 
 * copyright (c) 2021, Universidad Politécnica de Valencia (UPV), GAP research group
 * Date: July 2021
 * Author: GAP Research Group (UPV), contact: plopez@disca.upv.es
 * All rights reserved
 */


#include "eddl/mpi_distributed/mpi_distributed.h"

#ifdef cCUDA
#include "cuda.h"
#endif

#define NUM_STREAMS_COMM 1

// Global variables
int use_mpi = 0;
int mpi_avg = 1;
int avg_method = 0;
int x_avg = 0;
//int batch_is_global=1; 
// 1: Global batch=batch; Local batch=batch/n_procs 
// 0: Local batch=batch; Global_batch=batch*n_procs
int batches_avg = 0;
double secs_prev = 1E10;

#ifdef cNCCL
// NCCL
ncclUniqueId nccl_id;
ncclComm_t nccl_comm;
//cudaStream_t cuda_stream[NUM_STREAMS_COMM] ;
cudaStream_t cuda_stream;
#endif

int get_id_distributed() {
    int id = 0;
#ifdef cMPI
    //  Get the individual process ID.
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
#endif
    return id;
}

int get_n_procs_distributed() {
    int n_procs = 1;
#ifdef cMPI
    //  Get the number of processes.
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
#endif
    return n_procs;
}

int init_distributed() {
    int *argc;
    char ***argv;

    init_distributed(argc, argv);
}

int init_distributed(int *argc, char ***argv) {
    int id;
    int n_procs;

    id = init_MPI();
    //init_NCCL();
}

int init_MPI() {
    int *argc;
    char ***argv;

    int id;
    int n_procs;
    char node_name[256];
    int len;

#ifndef cMPI
    msg("Error: MPI library is not linked", "init_distributed");
#endif  

    id = 0;
#ifdef cMPI
    MPI_Init(argc, argv);

    use_mpi = 1;

    n_procs = get_n_procs_distributed();
    id = get_id_distributed();

    if (n_procs < 2) {
        msg("Error: Nr of MPI processes must be >1 ", "init_MPI");
    }

    MPICHECK(MPI_Get_processor_name(node_name, &len));
    fprintf(stderr, "[DISTR] MPI init. Node %d (%s). %d GPUS available\n", id, node_name, get_available_GPUs_distributed());


    fprintf(stderr, "[DISTR] setting default batch avg method\n");
    set_method_distributed(FIXED, AVG_DEFAULT, 0);

    // Initalize a different seed per proc
    srand(id * time(NULL));
#endif

    return id;
}

int init_NCCL(int nr_gpus) {
    int id;
    int n_procs;

    n_procs = get_n_procs_distributed();
    id = get_id_distributed();
#ifdef cNCCL
    //NCCL
    //get NCCL unique ID at rank 0 and broadcast it to all others
    if (id == 0) ncclGetUniqueId(&nccl_id);
    MPICHECK(MPI_Bcast(&nccl_id, sizeof (nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
    //picking a GPU based on localRank, allocate device buffers
    CUDACHECK(cudaSetDevice(id % nr_gpus));
    //for (int i = 0; i < NUM_STREAMS_COMM; i++) {
    //    CUDACHECK(cudaStreamCreateWithFlags(&cuda_stream[i], cudaStreamNonBlocking));
    //}
    CUDACHECK(cudaStreamCreate(&cuda_stream));
    //initializing NCCL
    NCCLCHECK(ncclCommInitRank(&nccl_comm, n_procs, nccl_id, id));
    if (id == 0)
        fprintf(stderr, "[DISTR] NCCL initialized %d procs\n", n_procs);
#endif

    return id;
}

void set_method_distributed(int method, int batch_avg, int epoch_avg) {

    int n_procs;
    int id;

#ifndef cMPI
    msg("MPI library is not linked", "set_method_distributed");
#endif  

    avg_method = method;
    mpi_avg = batch_avg;
    batches_avg = mpi_avg;
    x_avg = epoch_avg;

    n_procs = get_n_procs_distributed();
    id = get_id_distributed();

    if (id == 0)
        if (avg_method == FIXED) {
            fprintf(stderr, "[DISTR] method %s, batch_avg %d \n", "FIXED", mpi_avg);
        } else if (avg_method == AVG_INC) {
            fprintf(stderr, "[DISTR] method %s, batch_avg %d changing every %d epochs\n", "AVG_INC", mpi_avg, x_avg);
        } else if (avg_method == SAWTOOTH) {
            fprintf(stderr, "[DISTR] method %s, batch_avg %d changing every %d epochs\n", "SAWTOOTH", mpi_avg, x_avg);
        } else if (avg_method == NEG_SAWTOOTH) {
            fprintf(stderr, "[DISTR] method %s, batch_avg %d changing every %d epochs\n", "NEG SAWTOOTH", mpi_avg, x_avg);
        } else if (avg_method == AUTO_TIME) {
            fprintf(stderr, "[DISTR] method %s, batch_avg %d changing every %d epochs\n", "AUTO TIME", mpi_avg, x_avg);
        } else {
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
        MPI_Barrier(MPI_COMM_WORLD);
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

void fn_mpi_Broadcast(float* myptr, int count) {
#ifndef cMPI
    msg("invalid call to MPI_Allreduce. MPI library is not linked", "fn_mpi_AllReduce");
#endif

#ifdef cMPI
    if (count > 0) {
        MPICHECK(MPI_Bcast(myptr, count, MPI_FLOAT, 0, MPI_COMM_WORLD));
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

void fn_nccl_AllReduce_streams(float* myptr, int count, int layer) {
    int stream;
#ifdef cNCCL
    if (count > 0) {
        // TODO cuda_stream
        // stream= layer % NUM_STREAMS_COMM;
        //printf("Using stream %d\n", stream);
        NCCLCHECK(ncclAllReduce((const void*) myptr, (void*) myptr, count, ncclFloat, ncclSum, nccl_comm, cuda_stream));
        //completing NCCL operation by synchronizing on the CUDA stream
        CUDACHECK(cudaStreamSynchronize(cuda_stream));
    }
#endif
}

void fn_nccl_Broadcast_streams(float* myptr, int count, int layer) {
    int stream;
#ifdef cNCCL
    if (count > 0) {
        // TODO cuda_stream
        //stream= layer % NUM_STREAMS_COMM;
        //printf("Using stream %d\n", stream);
        NCCLCHECK(ncclBcast((void *) myptr, count, ncclFloat, 0, nccl_comm, cuda_stream));
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

void AllReduce_streams_distributed(float* myptr, int count, int layer) {
#ifdef cNCCL
    fn_nccl_AllReduce_streams(myptr, count, layer);
#else
    fn_mpi_AllReduce(myptr, count);
#endif
}

void Broadcast_streams_distributed(float* myptr, int count, int layer) {
#ifdef cNCCL
    fn_nccl_Broadcast_streams(myptr, count, layer);
#else
    fn_mpi_Broadcast(myptr, count);
#endif
}

int get_local_GPU_distributed(int id, int nGPUs) {
    int nDevices = 1;
    //#ifdef cCUDA
    //    cudaGetDeviceCount(&nDevices);
    //#endif
    //    return id % nDevices;
    return id % nGPUs;
}

int is_mpi_distributed() {
    return use_mpi;
}

int get_params_distributed(int* method, int* avg, int* avg_chg) {

    *avg = mpi_avg;
    *method = avg_method;
    *avg_chg = x_avg;

    return use_mpi;
}

int get_current_batch_avg_distributed() {
    return batches_avg;
}

int get_available_GPUs_distributed() {
    int count = 0;

    cudaGetDeviceCount(&count);
    return count;
}

void broadcast_CPU_params_distributed(Net* net) {
    int i, j;
    int root = 0;
    int size;

    vlayer layers = net->layers;
    for (i = 0; i < layers.size(); i++) {
        if (layers[i]->trainable) {
            for (j = 0; j < layers[i]->get_trainable_params_count(); j++) {


                float* myptr = layers[i]->params[j]->ptr;
                size = layers[i]->params[j]->size;
                fn_mpi_Broadcast(myptr, size);
            }
        }
    }
}

void broadcast_GPU_params_distributed(Net* net) {
    int i, j;
    int root = 0;
    int size;

    vlayer layers = net->snets[0]->layers;
    for (i = 0; i < layers.size(); i++) {
        if (layers[i]->trainable) {
            for (j = 0; j < layers[i]->get_trainable_params_count(); j++) {


                float* myptr = layers[i]->params[j]->ptr;
                size = layers[i]->params[j]->size;
                Broadcast_streams_distributed(myptr, size, i);
            }
        }
    }
}

void Bcast_params_distributed(Net * net) {
    broadcast_GPU_params_distributed(net);
}

void avg_weights_distributed(Net* net, int curr_batch, int batches_per_proc) {
    float * myptr;
    int count;
    int n_procs;


    int batches_avg;

    n_procs = get_n_procs_distributed();
    batches_avg = get_current_batch_avg_distributed();

    if ((((curr_batch) % batches_avg) == 0) || ((curr_batch) == batches_per_proc)) {
        //printf("Proc %d Sincronizando %d\n", id, j);
        for (int ii = 0; ii < net->snets[0]->layers.size(); ii++) {
            for (int jj = 0; jj < net->snets[0]->layers[ii]->params.size(); jj++) {

                myptr = net->snets[0]->layers[ii]->params[jj]->ptr;
                count = net->snets[0]->layers[ii]->params[jj]->size;
                //printf("\n===== Proc %d Batch %d Bucle ii=%d jj=%d size=%d\n", id, j, ii,jj,count );
                if (count != 0) {
                    // AllReduce params
                    //AllReduce_distributed(myptr, count);
                    AllReduce_streams_distributed(myptr, count, ii);
                    //fn_mpi_AllReduce(myptr, count)
                    //fn_nccl_AllReduce(myptr, count);

                    // Average params
                    net->snets[0]->layers[ii]->params[jj]->div_(n_procs);
                }
            }
        }
    }
}

void update_batch_avg_distributed(int epoch_id, double secs_epoch, int max_batch_avg) {
    float SPEED_UP = 1.05;
    switch (avg_method) {
        case AVG_INC:
            if (((epoch_id + 1) % (x_avg)) == 0) {
                if (batches_avg < max_batch_avg)
                    batches_avg = batches_avg * 2;
            }
            break;

        case SAWTOOTH:
            if (((epoch_id + 1) % (x_avg)) == 0) {
                batches_avg = batches_avg * 2;

                if (batches_avg >= max_batch_avg)
                    batches_avg = mpi_avg;
            }
            break;

        case NEG_SAWTOOTH:
            if (((epoch_id + 1) % (x_avg)) == 0) {
                batches_avg = batches_avg / 2;

                if (batches_avg < 1)
                    batches_avg = mpi_avg;
            }
            break;


        case AUTO_TIME:
            if (((epoch_id + 1) % (x_avg)) == 0) {
                float speed_up = secs_prev / secs_epoch;
                if (speed_up > SPEED_UP) {
                    secs_prev = secs_epoch;

                    if (batches_avg < max_batch_avg)
                        batches_avg = batches_avg * 2;
                }
            }
            break;
    }
}

void gpu_layer_print(Net* net, int ii) {
    printf("GPU tensor print. Layer %d \n", ii);

    float * cpu_buffer;
    float * myptr;
    int count;

    for (int jj = 0; jj < net->snets[0]->layers[ii]->params.size(); jj++) {
        myptr = net->snets[0]->layers[ii]->params[jj]->ptr;
        count = net->snets[0]->layers[ii]->params[jj]->size;
        cpu_buffer = (float *) malloc(count * sizeof (float));
        cudaMemcpy(cpu_buffer, myptr, count * sizeof (float), cudaMemcpyDeviceToHost);
        printf("Params: %d Size: %d\n", jj, count);
        int m = 0;
        for (int k = 0; k < count; k++) {
            printf("%7.4f, ", cpu_buffer[k]);
            m++;
            if ((m % 20) == 0)
                printf("\n");
        }
        printf("\n\n");
        free(cpu_buffer);
    }


}
