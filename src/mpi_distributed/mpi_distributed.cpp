/*
 * MPI support for EDDL Library - European Distributed Deep Learning Library.
 * Version: 
 * copyright (c) 2021, Universidad Politécnica de Valencia (UPV), GAP research group
 * Date: July 2021
 * Author: GAP Research Group (UPV), contact: plopez@disca.upv.es
 * All rights reserved
 */


#include "eddl/mpi_distributed/mpi_distributed.h"



#define GPU_1_distributed \
    switch (id % 1) { \
        case 0: gpus={1}; \
                gpu_str="1"; \
            break; \
        }  

#define GPU_2_distributed \
    switch (id % 2) { \
        case 0: gpus={1, 0}; \
                gpu_str="1,0"; \
            break; \
        case 1: gpus={0, 1}; \
                gpu_str="0,1"; \
            break; \
        }          

#define GPU_4_distributed \
    switch (id % 4) { \
        case 0: gpus={1, 0, 0, 0}; \
                gpu_str="1,0,0,0"; \
            break; \
        case 1: gpus={0, 1, 0, 0}; \
                gpu_str="0,1,0,0"; \
            break; \
        case 2: gpus={0, 0, 1, 0}; \
                gpu_str="0,0,1,0"; \
            break; \
        case 3: gpus={0, 0, 0, 1}; \
                gpu_str="0,0,0,1"; \
                          break; \
        }          

#define GPU_8_distributed \
    switch (id % 8) { \
        case 0: gpus={1, 0, 0, 0, 0, 0, 0, 0}; \
                gpu_str="1,0,0,0,0,0,0,0"; \
            break; \
        case 1: gpus={0, 1, 0, 0, 0, 0, 0, 0}; \
                gpu_str="0,1,0,0,0,0,0,0"; \
            break; \
        case 2: gpus={0, 0, 1, 0, 0, 0, 0, 0}; \
                gpu_str="0,0,1,0,0,0,0,0"; \
            break; \
        case 3: gpus={0, 0, 0, 1, 0, 0, 0, 0}; \
                gpu_str="0,0,0,1,0,0,0,0"; \
            break; \
        case 4: gpus={0, 0, 0, 0, 1, 0, 0, 0}; \
                gpu_str="0,0,0,0,1,0,0,0"; \
            break; \
        case 5: gpus={0, 0, 0, 0, 0, 1, 0, 0}; \
                gpu_str="0,0,0,0,0,1,0,0"; \
            break; \
        case 6: gpus={0, 0, 0, 0, 0, 0, 1, 0}; \
                gpu_str="0,0,0,0,0,0,1,0"; \
            break; \
        case 7: gpus={0, 0, 0, 0, 0, 0, 0, 1}; \
                gpu_str="0,0,0,0,0,0,0,1"; \
            break; \
        }         


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

float prev_losses=1e10;
float prev_metrics=0;

string lib;

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
    if (is_mpi_distributed())
        //  Get the individual process ID.
        MPI_Comm_rank(MPI_COMM_WORLD, &id);
#endif
    return id;
}

int get_n_procs_distributed() {
    int n_procs = 1;
#ifdef cMPI
    if (is_mpi_distributed())
        //  Get the number of processes.
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
#endif
    return n_procs;
}



void get_nodename_distributed(char* node_name) {
    int len;
#ifdef cMPI
    MPICHECK(MPI_Get_processor_name(node_name, &len));
#endif
}

int init_MPI() {
    int *argc;
    char ***argv;

    int id;
    int n_procs;
    char node_name[256] = "unknown";
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

    get_nodename_distributed(node_name);
    fprintf(stderr, "[DISTR] MPI init. Node %d (%s). %d GPUS available per node\n", id, node_name, get_available_GPUs_distributed());


    fprintf(stderr, "[DISTR] setting default batch avg method\n");
    set_method_distributed(FIXED, AVG_DEFAULT, 0);

    // Initalize a different seed per proc
    srand(id * time(NULL));
#endif

    return id;
}

void init_NCCL(int nr_gpus) {
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
        fprintf(stderr, "[DISTR] GPU: NCCL initialized %d procs\n", n_procs);
#endif
}

int init_distributed() {
    int id;

    id= init_distributed("NCCL");
    return id;
}

int init_distributed(string comm) {
    int id;
    int n_procs;

    id = init_MPI();
    if (comm == "NCCL") {
        lib = "NCCL";
        init_NCCL(get_available_GPUs_distributed());
    } else if (comm == "MPI") {
        lib = "MPI";
    } else {
        msg("Error unsupported communication library", "init_distributed"); // Exits
    }
    //fprintf(stderr, "[DISTR] using %s\n", lib.c_str());
    return id;
}

void end_distributed() {
    int id;
#ifndef cMPI
    msg("MPI library is not linked", "end_distributed");
#endif    


    if (use_mpi) {
#ifdef cMPI 
        MPI_Barrier(MPI_COMM_WORLD);
        //  Get the individual process ID.
        MPI_Comm_rank(MPI_COMM_WORLD, &id);
#endif

#ifdef cNCCL
        if (lib == "NCCL") {
            //finalizing NCCL
            ncclCommDestroy(nccl_comm);
            if (id == 0)
                fprintf(stdout, "[DISTR] GPU: NCCL End\n");
        }
#endif

#ifdef cMPI
        if (id == 0)
            fprintf(stdout, "[DISTR] End\n");
        MPI_Finalize();
#endif
    }
}

int is_mpi_distributed() {
    return use_mpi;
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
            msg("Error avg_method", "set_method_distributed"); // Exits
            //fprintf(stderr, "[DISTR] Error avg method %d not implemented\n", avg_method);
            //exit(EXIT_FAILURE);
        }
}

vector<int> get_gpu_vec_distributed() {
    int id;
    vector<int> gpus;
    string gpu_str;
    char node_name[256] = "unknown";

    int nr_gpus = get_available_GPUs_distributed();

    if (is_mpi_distributed()) {
        id = get_id_distributed();
    } else {
        id = 0;
    }

    switch (nr_gpus) {
        case 1: GPU_1_distributed;
            break;
        case 2: GPU_2_distributed;
            break;
        case 4: GPU_4_distributed;
            break;
        case 8: GPU_8_distributed;
            break;
        default: msg("Error nr_gpus param", "mpi_distributed CS_GPU()"); // Exits
    }

    if (is_mpi_distributed()) {
        get_nodename_distributed(node_name);
        fprintf(stderr, "[DISTR] Node: %s. Process %d. CS: GPU mask: %s\n", node_name, id, gpu_str.c_str());
    } else {
        fprintf(stderr, "[CS_GPU()] CS: GPU mask: %s\n", gpu_str.c_str());
    }


    return gpus;
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
#ifdef cGPU
    cudaGetDeviceCount(&count);
#endif
    return count;
}

void fn_mpi_AllReduce(float* myptr, int count) {


#ifdef cMPI
    if (count > 0) {
        MPICHECK(MPI_Allreduce(MPI_IN_PLACE, myptr, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
    }
#else
    msg("invalid call. MPI library is not linked", "fn_mpi_AllReduce");
#endif
}

void fn_mpi_Bcast(float* myptr, int count) {
#ifdef cMPI
    if (count > 0) {
        MPICHECK(MPI_Bcast(myptr, count, MPI_FLOAT, 0, MPI_COMM_WORLD));
        printf("======fn_mpi_Bcast\n");
    }
#else
    msg("invalid call. MPI library is not linked", "fn_mpi_Broadcast");
#endif
}

void fn_nccl_AllReduce(float* myptr, int count) {
#ifdef cNCCL
    if (count > 0) {
        NCCLCHECK(ncclAllReduce((const void*) myptr, (void*) myptr, count, ncclFloat, ncclSum, nccl_comm, cuda_stream));
        //completing NCCL operation by synchronizing on the CUDA stream
        CUDACHECK(cudaStreamSynchronize(cuda_stream));
    }
#else
    msg("invalid call. NCCL library is not linked", "fn_nccl_Allreduce");
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
#else
    msg("invalid call. NCCL library is not linked", "fn_nccl_AllReduce_streams");
#endif
}

void fn_nccl_Bcast(float* myptr, int count) {
#ifdef cNCCL
    if (count > 0) {
        NCCLCHECK(ncclBcast((void *) myptr, count, ncclFloat, 0, nccl_comm, cuda_stream));
        //completing NCCL operation by synchronizing on the CUDA stream
        CUDACHECK(cudaStreamSynchronize(cuda_stream));
    }
#else
    msg("invalid call. NCCL library is not linked", "fn_nccl_Broadcast");
#endif
}

void fn_nccl_Bcast_streams(float* myptr, int count, int layer) {
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
#else
    msg("invalid call. NCCL library is not linked", "fn_nccl_Broadcast_streams");
#endif
}

void fn_GPU_AllReduce(float* myptr, int count) {
    if (lib == "NCCL") {
        fn_nccl_AllReduce(myptr, count);
    } else {
        fn_mpi_AllReduce(myptr, count);
    }
}

int get_local_GPU_distributed(int id, int nGPUs) {
    int nDevices = 1;
    //#ifdef cGPU
    //    cudaGetDeviceCount(&nDevices);
    //#endif
    //    return id % nDevices;
    return id % nGPUs;
}

void fn_Bcast_CPU_weights(Net* net) {
    int i, j;
    int root = 0;
    int size;

    vlayer layers = net->layers;
    for (i = 0; i < layers.size(); i++) {
        if (layers[i]->trainable) {
            for (j = 0; j < layers[i]->get_trainable_params_count(); j++) {
                float* myptr = layers[i]->params[j]->ptr;
                size = layers[i]->params[j]->size;
                fn_mpi_Bcast(myptr, size);
            }
        }
    }
}

void fn_Bcast_GPU_weights(Net* net) {
    float * myptr;
    int count;
    int i, j;


    for (int i = 0; i < net->layers.size(); i++) {
        if (net->layers[i]->trainable) {
            for (int j = 0; j < net->layers[i]->get_trainable_params_count(); j++) {
                myptr = net->snets[0]->layers[i]->params[j]->ptr;
                count = net->snets[0]->layers[i]->params[j]->size;
                if (lib == "NCCL")
                    fn_nccl_Bcast(myptr, count);
                else
                    fn_mpi_Bcast(myptr, count);
            }
        }
    }
}

void bcast_weights_distributed(Net * net) {
    if (net->cs->hw == "gpu")
        fn_Bcast_GPU_weights(net);
    else if (net->cs->hw == "cpu")
        fn_Bcast_CPU_weights(net);
    else
        msg("Error unsupported device", "bcast_params_distributed"); // Exits
}

void avg_GPU_weights_distributed(Net* net, int curr_batch, int batches_per_proc) {
    float * myptr;
    int count;
    int n_procs;


    int batches_avg;

    n_procs = get_n_procs_distributed();
    batches_avg = get_current_batch_avg_distributed();

    if (((curr_batch % batches_avg) == 0) || (curr_batch == batches_per_proc)) {
        //printf("Proc %d Sincronizando %d\n", id, j);
        for (int i = 0; i < net->layers.size(); i++) {
            if (net->layers[i]->trainable) {
                for (int j = 0; j < net->layers[i]->get_trainable_params_count(); j++) {
                    //for (int ii = 0; ii < net->snets[0]->layers.size(); ii++) {
                    //    for (int jj = 0; jj < net->snets[0]->layers[ii]->params.size(); jj++) {                
                    myptr = net->snets[0]->layers[i]->params[j]->ptr;
                    count = net->snets[0]->layers[i]->params[j]->size;
                    //printf("\n===== Proc %d Batch %d Bucle ii=%d jj=%d size=%d\n", id, j, ii,jj,count );
                    if (count != 0) {
                        // AllReduce params
                        fn_GPU_AllReduce(myptr, count);
                        // Average params
                        net->snets[0]->layers[i]->params[j]->div_(n_procs);
                    }
                }
            }
        }
    }
}

void avg_CPU_weights_distributed(Net* net, int curr_batch, int batches_per_proc) {
    float * myptr;
    int count;
    int n_procs;


    int batches_avg;

    n_procs = get_n_procs_distributed();
    batches_avg = get_current_batch_avg_distributed();

    if ((((curr_batch) % batches_avg) == 0) || ((curr_batch) == batches_per_proc)) {
        //printf("Proc %d Sincronizando %d\n", id, j);
        for (int i = 0; i < net->layers.size(); i++) {
            if (net->layers[i]->trainable) {
                for (int j = 0; j < net->layers[i]->get_trainable_params_count(); j++) {
                    //for (int ii = 0; ii < net->snets[0]->layers.size(); ii++) {
                    //    for (int jj = 0; jj < net->snets[0]->layers[ii]->params.size(); jj++) {
                    myptr = net->layers[i]->params[j]->ptr;
                    count = net->layers[i]->params[j]->size;
                    //printf("\n===== Proc %d Batch %d Bucle ii=%d jj=%d size=%d\n", id, j, ii,jj,count );
                    if (count != 0) {
                        // AllReduce params
                        fn_mpi_AllReduce(myptr, count);
                        // Average params
                        net->layers[i]->params[j]->div_(n_procs);
                    }
                }
            }
        }
    }
}

void avg_weights_distributed(Net* net, int curr_batch, int batches_per_proc) {
    if (net->cs->hw == "gpu")
        avg_GPU_weights_distributed(net, curr_batch, batches_per_proc);
    else if (net->cs->hw == "cpu")
        avg_CPU_weights_distributed(net, curr_batch, batches_per_proc);
    else
        msg("Error unsupported device", "avg_weights_distributed"); // Exits
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

#ifdef cGPU
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
#else
    printf("Error: CUDA is not available\n");
#endif

}

bool early_stopping_on_loss_var(Net* net, int index, float delta, int patience, int epoch) {
    int id = get_id_distributed();
    float losses = net->get_losses()[index];
    bool result;

    if (id == 0)
        if (epoch > patience) {
                printf("[DISTR] prev_loss: %f, loss: %f\n", prev_losses, losses);
            if (losses > (delta+prev_losses)) {
                printf("[DISTR] Early Stopping! (prev_loss: %f, loss: %f)\n", prev_losses, losses);
                result = true;
            } else {
                result = false;
            }
            if (losses < prev_losses) {
                printf("[DISTR] Early Stopping! Epoch %d. Best loss (prev_loss: %f, loss: %f)\n", epoch, prev_losses, losses);
               prev_losses = losses;            
            }
        } else 
            result = false;
    
    if (is_mpi_distributed())
#ifdef cMPI       
        MPICHECK(MPI_Bcast(&result, 1, MPI_BYTE, 0, MPI_COMM_WORLD));
#endif    
    return result;
}

bool early_stopping_on_metric_var(Net* net, int index, float delta, int patience, int epoch) {
    int id = get_id_distributed();
    float metrics = net->get_metrics()[index];
    bool result;

    if (id == 0)
        if (epoch > patience) {
            if (metrics > prev_metrics) {// OK
                if ((metrics - prev_metrics) < delta) {
                    printf("[DISTR] Early Stopping! ((metric %f-prev_metric %f) < delta %f)\n", metrics, prev_metrics, delta);
                    result = true;
                } else {
                    prev_metrics = metrics;
                    result = false;
                }
            } else if ((prev_metrics - metrics) > delta) {
                    printf("[DISTR] Early Stopping! ((prev_metric %f-metric %f) < delta %f)\n", prev_metrics, metrics, delta);
                    result = true;
                } else {
                    result = false;
                }
        } else
            result = false;

    if (is_mpi_distributed())
#ifdef cMPI
        MPICHECK(MPI_Bcast(&result, 1, MPI_BYTE, 0, MPI_COMM_WORLD));
#endif    
    return result;
}

bool early_stopping_on_metric(Net* net, int index, float goal, int patience, int epoch) {
    int id=get_id_distributed();
    float metrics = net->get_metrics()[index];
    bool result;

    if (id == 0)
        if (epoch > patience) {
            if (metrics > goal) {// OK
                printf("[DISTR] Early Stopping! ((metric %f >goal %f)\n",metrics,goal);
                result = true;
            } else {
                result = false;
            }
        } else {
            result = false;
        }
    
    if (is_mpi_distributed())
#ifdef cMPI
        MPICHECK(MPI_Bcast(&result, 1, MPI_BYTE, 0, MPI_COMM_WORLD));
#endif
    return result;
}

float quantize(float value, int nbits_int, int nbits_frac) {
    float result = 0;
    int i;
    int bit;

    // We convert the value to an integer, no frac part
    int x = round(value * pow(2, nbits_frac));
    int maxint = pow(2, (nbits_int + nbits_frac)) - 1;

    if (x >= maxint) {
        result = maxint;
    } else
        if (x <= -maxint) {
        result = -maxint;
    } else {
        i = 0;
        while (i < (nbits_int + nbits_frac)) {
            bit = x % 2;
            x = x / 2;
            result = result + bit * pow(2, i);
            //printf("Int ... %f", result);
            i++;
        }
    }
    result = result / pow(2, nbits_frac);

    return (result);
}

void quantize_network_distributed(Net* net, int nbits_int, int nbits_frac) {
    float * myptr;
    int count;
    int n_procs;

    for (int i = 0; i < net->layers.size(); i++) {
        if (net->layers[i]->trainable) {
            for (int j = 0; j < net->layers[i]->get_trainable_params_count(); j++) {  
                //printf("\n===== Proc %d Batch %d Bucle ii=%d jj=%d size=%d\n", id, j, ii,jj,count );
                // copy from devices
                for (int dev = 0; dev < net->snets.size(); dev++) {
                    Tensor::copy(net->snets[dev]->layers[i]->params[j], net->layers[i]->params[j]);
                }
                myptr = net->layers[i]->params[j]->ptr;
                count = net->layers[i]->params[j]->size;
                for (int k = 0; k < count; k++) {
                    //printf("quantize: %f %f \n", myptr[k], quantize(myptr[k], nbits_int, nbits_frac));
                    myptr[k] = quantize(myptr[k], nbits_int, nbits_frac);
                }
                // copy-back to devices
                for (int dev = 0; dev < net->snets.size(); dev++) {
                    Tensor::copy(net->layers[i]->params[j], net->snets[dev]->layers[i]->params[j]);
                }
            }
        }
    }
    
}

