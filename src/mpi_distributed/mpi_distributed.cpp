/*
 * MPI support for EDDL Library - European Distributed Deep Learning Library.
 * Version: 
 * copyright (c) 2021, Universidad Politécnica de Valencia (UPV), GAP research group
 * Date: July 2021
 * Author: GAP Research Group (UPV), contact: plopez@disca.upv.es
 * All rights reserved
 */

#define _FILE_OFFSET_BITS 64

#include "eddl/mpi_distributed/mpi_distributed.h"

#include <sys/types.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>


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
int id=0;
int n_procs=1;
int mpi_avg = 1;
int avg_method = 0;
int x_avg = 1;
float comm_overhead = 0.1;
//int batch_is_global=1; 
// 1: Global batch=batch; Local batch=batch/n_procs 
// 0: Local batch=batch; Global_batch=batch*n_procs
int batches_avg = 0;
double secs_prev = 1E10;
int cuda_aware_bcast=0; // Default: do not use Cuda aware Bcast
int cuda_aware_allreduce=1; // Default: use Cuda aware AllReduce

float prev_losses=1e10;
float prev_metrics=0;

#define SILENT 1

#define check_MPI(action) \
    if (n_procs==1) { \
        if (!SILENT) \
            fprintf(stderr,"[DISTR] Warning. Distributed mode is off. Call to %s\n", __func__); \
        action; \
    } 


string lib;

#ifdef cNCCL
// NCCL
ncclUniqueId nccl_id;
ncclComm_t nccl_comm;
//cudaStream_t cuda_stream[NUM_STREAMS_COMM] ;
cudaStream_t cuda_stream;
#endif


#define MAX_BUFFER 1024
#define MAX_DG_THREADS 32
int DEBUG=0;


FILE* fpX;
FILE* fpY;
int buffer_count=0;
int ptr_in=0;
int ptr_out=0;
int ds_ptr=0;
sem_t dmutex;
sem_t llenar;
sem_t vaciar;
sem_t imprimir;
pthread_t t[MAX_DG_THREADS];

int dataset_size;
int ndimX;
int ndimY;
int shape_sizeX;
int shape_sizeY;

int dg_batch_size=0;
int dg_num_batches;
int dg_buffer_size=0;
int dg_num_threads;
float* bufferX;
float* bufferY;
int* list;
bool dg_perfect=false;
bool first_epoch=true;
char tmp_name[128]="/tmp/eddl_dg.txt";
 FILE* tmp_fp;
// For debugging purposes
//#define DEBUG_DONE 
size_t n_sizeX;
size_t n_sizeY;
 unsigned char* bytesX;
 unsigned char* bytesY;
int* done_batches;
int* done_images;

struct thread_info {    /* Used as argument to thread_start() */
           pthread_t thread_id;        /* ID returned by pthread_create() */
           int       thread_num;       /* Application-defined thread # */
           char     *argv_string;      /* From command-line argument */
       };

void check_MPI_Cuda_Aware() {
    printf("[DISTR] Compile-time MPI CUDA Aware check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    printf("[DISTR] This MPI library has CUDA-aware support.\n", MPIX_CUDA_AWARE_SUPPORT);
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
    printf("[DISTR] This MPI library does not have CUDA-aware support.\n");
#else
    printf("[DISTR] This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */
 
    printf("[DISTR] Runtime MPI CUDA Aware check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    if (1 == MPIX_Query_cuda_support()) {
        printf("[DISTR] This MPI library has CUDA-aware support.\n");
    } else {
        printf("[DISTR] This MPI library does not have CUDA-aware support.\n");
    }
#else /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
    printf("[DISTR] This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */
}

int get_id_distributed() {
    /**
    int id = 0;
#ifdef cMPI
    if (is_mpi_distributed()) {
        //  Get the individual process ID.
        MPI_Comm_rank(MPI_COMM_WORLD, &id);
    }
#endif
     */
    return id;
}

int get_n_procs_distributed() {
    /*
    int n_procs = 1;
#ifdef cMPI
    if (is_mpi_distributed()) {
        //  Get the number of processes.
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
     }
#endif
     */
    return n_procs;
}

void get_nodename_distributed(char* node_name) {
    int len;
#ifdef cMPI
    MPICHECK(MPI_Get_processor_name(node_name, &len));
#endif
}

int init_MPI(int *argc, char ***argv) {
    char node_name[256] = "unknown";
    int len;
    int provided=1;

#ifndef cMPI
    msg("Error: MPI library is not linked", "init_distributed");
#endif  

    id = 0;
#ifdef cMPI
    //MPI_Init(argc, argv);
    MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
    //MPI_Init_thread(argc, argv, MPI_THREAD_SERIALIZED, &provided);
    //MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided==0) 
        msg("Error multiple threads not supported in MPI library", "init_MPI"); // Exits
    //fprintf(stdout, "[DISTR] MPI init. %d multiple threads\n", id);


    use_mpi = 1;

    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    check_MPI();
    /*
    if (n_procs < 2) {
        msg("Error: Nr of MPI processes must be >1 ", "init_MPI");
    }
    */

    get_nodename_distributed(node_name);
    fprintf(stdout, "[DISTR] init_MPI. Node %d/%d (%s). %d GPUS available per node\n", id, n_procs, node_name, get_available_GPUs_distributed());

    if (id==0)
        fprintf(stdout, "[DISTR] setting default batch avg method\n");
    set_avg_method_distributed(FIXED, AVG_DEFAULT);

    // Initalize a different seed per proc
    srand((id+1) * time(NULL));
#endif
    return id;
}

void init_NCCL(int nr_gpus) {

#ifdef cNCCL
    //NCCL
    //get NCCL unique ID at rank 0 and broadcast it to all others
    if (id == 0) ncclGetUniqueId(&nccl_id);
#ifdef cMPI
    MPICHECK(MPI_Bcast(&nccl_id, sizeof (nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
#endif
    //picking a GPU based on localRank, allocate device buffers
#ifdef cGPU
    CUDACHECK(cudaSetDevice(id % nr_gpus));
    //for (int i = 0; i < NUM_STREAMS_COMM; i++) {
    //    CUDACHECK(cudaStreamCreateWithFlags(&cuda_stream[i], cudaStreamNonBlocking));
    //}
    CUDACHECK(cudaStreamCreate(&cuda_stream));
#endif
    //initializing NCCL
    NCCLCHECK(ncclCommInitRank(&nccl_comm, n_procs, nccl_id, id));
    if (id == 0)
        fprintf(stdout, "[DISTR] GPU: NCCL initialized %d procs\n", n_procs);
#else
    msg("invalid call. NCCL library is not linked", __func__);
#endif
}

int init_distributed() {
    int id;

    
#ifdef cNCCL
    id= init_distributed("NCCL");
#else  
    fprintf(stdout, "[DISTR] %s. NCCL not available. Using MPI\n", __func__);
    id= init_distributed("MPI");  
#endif
    return id;
}

int init_distributed(string comm) {
    int id;
    int n_procs;
    int *argc;
    char ***argv;

    id = init_MPI(argc, argv);
    barrier_distributed();
   
    if (comm == "NCCL") {
        lib = "NCCL";
        init_NCCL(get_available_GPUs_distributed());
    } else if (comm == "MPI") {
        lib = "MPI";
    } else if (comm == "MPI-NCA") {
        lib = "MPI-Non CUDA aware";
        cuda_aware_allreduce=0;    
    } else {
       msg("Error unsupported communication library", __func__); // Exits
    }

    if (id == 0) {
        fprintf(stdout, "[DISTR] %s. lib=%s\n", __func__, lib.c_str());
        check_MPI_Cuda_Aware();
        if (cuda_aware_allreduce)
            fprintf(stdout, "[DISTR] %s. ENABLED cuda aware AllReduce\n", __func__);
        else
            fprintf(stdout, "[DISTR] %s. DISBLED cuda aware AllReduce \n", __func__);
    }
    barrier_distributed();
    //fprintf(stdout, "[DISTR] using %s\n", lib.c_str());
    return id;
}

/*
int init_distributed2(int *argc, char ***argv) {
    //int id;

    id= init_distributed2(argc, argv, "NCCL");
    return id;
}

int init_distributed2(int *argc, char ***argv, string comm) {
 
    id = init_MPI(argc,argv);
    if (comm == "NCCL") {
        lib = "NCCL";
        init_NCCL(get_available_GPUs_distributed());
    } else if (comm == "MPI") {
        lib = "MPI";
    } else {
        msg("Error unsupported communication library", "init_distributed"); // Exits
    }
    //fprintf(stdout, "[DISTR] using %s\n", lib.c_str());
    return id;
}
*/

void end_distributed() {
    //int id;
#ifndef cMPI
    msg("MPI library is not linked", __func__);
#endif    
    if (!use_mpi) {
        return;
    }
#ifdef cMPI 
    MPI_Barrier(MPI_COMM_WORLD);
    //  Get the individual process ID.
    //MPI_Comm_rank(MPI_COMM_WORLD, &id);
#endif

#ifdef cNCCL
    if (lib == "NCCL") {
        //finalizing NCCL
        ncclCommDestroy(nccl_comm);
        if (id == 0)
            fprintf(stdout, "[DISTR] %s: NCCL End\n", __func__);
    }
#endif

#ifdef cMPI
    if (id == 0)
        fprintf(stdout, "[DISTR] End\n");
    MPI_Finalize();
#endif

}

int is_mpi_distributed() {
    return use_mpi;
}

void set_avg_method_distributed(int method, int batch_avg, int epoch_avg, float overhead) {

    //int n_procs;
    //int id;

#ifndef cMPI
    msg("MPI library is not linked", __func__);
#endif  
    check_MPI();
    
    avg_method = method;
    mpi_avg = batch_avg;
    batches_avg = mpi_avg;
    x_avg = epoch_avg;
    comm_overhead = overhead;

    //n_procs = get_n_procs_distributed();
    //id = get_id_distributed();

    if (id == 0)
        if (avg_method == FIXED) {
            fprintf(stdout, "[DISTR] %s %s, batch_avg %d \n", __func__, "FIXED", mpi_avg);
        } else if (avg_method == AVG_INC) {
            fprintf(stdout, "[DISTR] %s %s, batch_avg %d changing every %d epochs\n", __func__, "AVG_INC", mpi_avg, x_avg);
        } else if (avg_method == SAWTOOTH) {
            fprintf(stdout, "[DISTR] %s %s, batch_avg %d changing every %d epochs\n", __func__, "SAWTOOTH", mpi_avg, x_avg);
        } else if (avg_method == NEG_SAWTOOTH) {
            fprintf(stdout, "[DISTR] %s %s, batch_avg %d changing every %d epochs\n", __func__, "NEG SAWTOOTH", mpi_avg, x_avg);
        } else if (avg_method == AUTO_TIME) {
            fprintf(stdout, "[DISTR] %s %s, batch_avg %d changing every %d epochs\n", __func__, "AUTO TIME", mpi_avg, x_avg);
        } else if (avg_method == LIMIT_OVERHEAD) {
            fprintf(stdout, "[DISTR] %s %s, batch_avg %d comm. overhead %2.1f\n", __func__, "LIMIT_OVERHEAD", mpi_avg, comm_overhead);
        } else {
            msg("Error unknown avg_method",  __func__); // Exits
        }
    barrier_distributed();
}

vector<int> get_gpu_vec_distributed() {
    int id;
    vector<int> gpus;
    string gpu_str;
    char node_name[256] = "unknown";

    check_MPI();

    
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
        fprintf(stdout, "[DISTR] get_gpu_vec. Node: %s. Process %d. CS: GPU mask: %s\n", node_name, id, gpu_str.c_str());
        fprintf(stdout, "[DISTR] EDDL DeviceID %d \n", Tensor::getDeviceID("cuda"));
    } else {
        fprintf(stdout, "[CS_GPU()] CS: GPU mask: %s\n", gpu_str.c_str());
    }

    barrier_distributed();
    return gpus;
}

int get_params_distributed(int* method, int* avg, int* avg_chg) {
    
    check_MPI();


    *avg = mpi_avg;
    *method = avg_method;
    *avg_chg = x_avg;

    return use_mpi;
}

int get_avg_method_distributed() {
    return (avg_method);
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

int get_available_CPUs_distributed() {
    return omp_get_num_procs();
}

void set_OMP_threads_to_procs_distributed() {
    omp_set_num_threads(omp_get_num_procs());
}


void set_batch_distributed (int* global_batch, int* local_batch, int batch, int method) {
   //int id;
   // int n_procs;
    
    check_MPI();
    //id = get_id_distributed();
    //n_procs = get_n_procs_distributed();  
    if (method == DIV_BATCH) {
        *global_batch=batch;
        *local_batch=batch/n_procs;
    } else if (method == MUL_BATCH) {
        *global_batch=batch*n_procs;
        *local_batch=batch;
    } else {
         msg("Error batch distributed method", __func__); // Exits
    }   
    if (id==0)
        printf("[DISTR] set_batch. Method: %s. Batch size: %d. Global batch size: %d. Local batch size:  %d\n",  (method==DIV_BATCH?"DIV":"MUL"), batch, *global_batch, *local_batch);
    barrier_distributed();
    
    return;
}

int set_NBPP_distributed(int ds_size, int local_batch, bool method) {
   // int id;
   // int n_procs;
    int num_batches;
    int nbpp;

    check_MPI();
    //id = get_id_distributed();
    //n_procs = get_n_procs_distributed();  
    num_batches = ds_size / local_batch;
    if (method == NO_DISTR_DS) {
        nbpp = num_batches / n_procs;
    } else if (method == DISTR_DS) {
        nbpp = num_batches;
    } else {
        msg("Error num_batches_per_proc method",  __func__); // Exits
    }
    if (id==0)
        printf("[DISTR] set_NBPP. Proc: %d. Distr dataset: %s. Dataset size: %d. Local batch: %d. Num batches per proc: %d\n", id, (method==NO_DISTR_DS?"no":"yes"), ds_size, local_batch, nbpp);
   
    barrier_distributed();
    
    return nbpp;
}

void fn_mpi_AllReduce(float* myptr, int count) {
#ifdef cMPI
    if (count > 0) {
        MPICHECK(MPI_Allreduce(MPI_IN_PLACE, myptr, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
    }
#else
    msg("invalid call. MPI library is not linked",  __func__);
#endif
}

void fn_mpi_Bcast(float* myptr, int count) {
#ifdef cMPI
    if (count > 0) {
        MPICHECK(MPI_Bcast(myptr, count, MPI_FLOAT, 0, MPI_COMM_WORLD));
        //printf("======fn_mpi_Bcast\n");
    }
#else
    msg("invalid call. MPI library is not linked",  __func__);
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
    msg("invalid call. NCCL library is not linked",  __func__);
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
    msg("invalid call. NCCL library is not linked",  __func__);
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
    msg("invalid call. NCCL library is not linked",  __func__);
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
    msg("invalid call. NCCL library is not linked",  __func__);
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
                else {
                    if (cuda_aware_bcast == 0) {
                        // Non CUDA-aware version
                        // CUDA-aware Bcast cause segmentation fault
                        //printf("############ BROADCAST ############\n");
                        Tensor::copy(net->snets[0]->layers[i]->params[j], net->layers[i]->params[j]);
                        fn_mpi_Bcast(net->layers[i]->params[j]->ptr, count);
                        Tensor::copy(net->layers[i]->params[j], net->snets[0]->layers[i]->params[j]);
                    } else {
                        fn_mpi_Bcast(myptr, count);
                    }
                }
            }
        }
    }
}

void bcast_weights_distributed(Net * net) {
    check_MPI(return);

    if (id==0)
        printf("[DISTR] %s.\n", __func__);
   
    if (net->cs->hw == "gpu")
        fn_Bcast_GPU_weights(net);
    else if (net->cs->hw == "cpu")
        fn_Bcast_CPU_weights(net);
    else
        msg("Error unsupported device",  __func__); // Exits
}

void avg_GPU_weights_distributed(Net* net, int curr_batch, int batches_per_proc) {
    float * myptr;
    int count;
   // int n_procs;
   // int id;
    //int batches_avg;
   // id = get_id_distributed();
   // n_procs = get_n_procs_distributed();
    //batches_avg = get_current_batch_avg_distributed();

    if (((curr_batch % batches_avg) == 0) || (curr_batch == batches_per_proc)) {
        //printf("Proc %d Sincronizando batch nr %d bpp %d\n", id, curr_batch, batches_per_proc);
        for (int i = 0; i < net->layers.size(); i++) {
            if (net->layers[i]->trainable) {
                for (int j = 0; j < net->layers[i]->get_trainable_params_count(); j++) {
                    //for (int ii = 0; ii < net->snets[0]->layers.size(); ii++) {
                    //    for (int jj = 0; jj < net->snets[0]->layers[ii]->params.size(); jj++) {                
                    myptr = net->snets[0]->layers[i]->params[j]->ptr;
                    count = net->snets[0]->layers[i]->params[j]->size;
                    //printf("\n===== Proc %d Batch %d Bucle i=%d j=%d size=%d\n", id, curr_batch, i,j,count );
                    if (count != 0) {
                        // AllReduce params
                        if (lib == "NCCL") {
                            fn_nccl_AllReduce(myptr, count);
                        } else {
                            if (cuda_aware_allreduce == 0) {
                                // Non CUDA-aware version
                                // CUDA-aware Bcast cause segmentation fault
                                //printf("############ BROADCAST ############\n");
                                Tensor::copy(net->snets[0]->layers[i]->params[j], net->layers[i]->params[j]);
                                fn_mpi_AllReduce(net->layers[i]->params[j]->ptr, count);
                                Tensor::copy(net->layers[i]->params[j], net->snets[0]->layers[i]->params[j]);
                            } else {
                                fn_mpi_AllReduce(myptr, count);
                            }
                        }
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
   // int n_procs;
    //int batches_avg;
    //n_procs = get_n_procs_distributed();
    //batches_avg = get_current_batch_avg_distributed();

    if ((((curr_batch) % batches_avg) == 0) || ((curr_batch) == batches_per_proc)) {
       // printf("Proc %d Sincronizando \n", id);
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

void avg_metrics_distributed(Net* net) {
    //int n_procs;
    //n_procs = get_n_procs_distributed();
    check_MPI(return);

    for (int k = 0; k < net->lout.size(); k += net->decsize) {
        if (net->losses.size() >= (k + 1)) {
#ifdef cMPI       
            MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &net->total_loss[k], 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
            net->total_loss[k] = net->total_loss[k] / n_procs;
#endif
        }
        if (net->metrics.size() >= (k + 1)) {
#ifdef cMPI                                     
            MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &net->total_metric[k], 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
            net->total_metric[k] = net->total_metric[k] / n_procs;
#endif                    
        }
    }
}

void avg_float_distributed(float* pvar) {
    //int n_procs;
    //n_procs = get_n_procs_distributed();
    //printf ("== Funcion %s\n", __func__);
    check_MPI(return)

#ifdef cMPI       
        MPICHECK(MPI_Allreduce(MPI_IN_PLACE, pvar, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
        *pvar = *pvar / n_procs;  
#endif
}

void barrier_distributed() {
    //int n_procs;
    //n_procs = get_n_procs_distributed();
    check_MPI(return);

    if (is_mpi_distributed()) {
#ifdef cMPI       
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
#endif
    }
}

void avg_weights_distributed(Net* net, int curr_batch, int batches_per_proc) {
    check_MPI(return);
    if (net->cs->hw == "gpu")
        avg_GPU_weights_distributed(net, curr_batch, batches_per_proc);
    else if (net->cs->hw == "cpu")
        avg_CPU_weights_distributed(net, curr_batch, batches_per_proc);
    else
        msg("Error unsupported device", __func__); // Exits
}

void update_batch_avg_distributed(int epoch_id, double secs_epoch, int max_batch_avg) {
    float TH_UP = 1.10;
    float TH_DN = 0.97;
    int prev_ba=batches_avg;

    check_MPI(return);
     
    if (id == 0) {
        switch (avg_method) {
            case AVG_INC:
                if (((epoch_id + 1) % (x_avg)) == 0) {
                    batches_avg = std::min(batches_avg * 2, max_batch_avg);
                    printf("[DISTR] method AVG_INC,batches_avg %d -->  %d \n", prev_ba, batches_avg);
                }
                break;

            case SAWTOOTH:
                if (((epoch_id + 1) % (x_avg)) == 0) {
                    batches_avg = std::min(batches_avg * 2, max_batch_avg);
                    printf("[DISTR] method SAWTOOTH, batches_avg %d -->  %d \n", prev_ba, batches_avg);
                }
                break;

            case NEG_SAWTOOTH:
                if (((epoch_id + 1) % (x_avg)) == 0) {
                    batches_avg = std::max(batches_avg / 2, mpi_avg);
                    printf("[DISTR] method NEG_SAWTOOTH, batches_avg %d -->  %d \n", prev_ba, batches_avg);
                }
                break;

            case AUTO_TIME:
                if (((epoch_id + 1) % (x_avg)) == 0) {
                    float speed_up = secs_prev / secs_epoch;
                   
                    if (speed_up > TH_UP) { // OK, let's reduce comm
                        //printf("Mayor %f %f\n", secs_prev, speed_up);
                        if (secs_epoch<secs_prev)
                            secs_prev = secs_epoch;
                        batches_avg = std::min(batches_avg * 2, max_batch_avg);
                    } else if (speed_up < TH_DN) { // OOPs, train time increased
                        //printf("Menor %f %f\n", secs_prev,speed_up);
                        secs_prev = secs_epoch; // Reset reference
                        batches_avg = std::min(batches_avg * 2, max_batch_avg);
                    } else { // Don't do anything
                        //printf("Medio %f %f\n", secs_prev,speed_up);
                        if (secs_epoch<secs_prev)
                            secs_prev = secs_epoch;
                        batches_avg = std::max(batches_avg - (batches_avg / 4), mpi_avg);
                    }
                    printf("[DISTR] method AUTO_TIME, batches_avg %d -->  %d \n", prev_ba, batches_avg);
                }
                break;
            default:
                //mpi_id0(printf("[DISTR] %s: batches_avg unchanged\n",__func__));
                break;
        }
    }
#ifdef cMPI
    MPICHECK(MPI_Bcast(&batches_avg, 1, MPI_INT, 0, MPI_COMM_WORLD));
#endif  
}

void set_batch_avg_overhead_distributed(double secs_train, double secs_comm, int max_ba) {
    set_batch_avg_overhead_distributed(secs_train, secs_comm, max_ba, comm_overhead);
}


void set_batch_avg_overhead_distributed(double secs_train, double secs_comm, int max_ba, float overhead)  {
    check_MPI(return);
    double comm1;
    double ba;
    int prev_ba=batches_avg;
    int new_ba;

    if (avg_method == LIMIT_OVERHEAD) {
        if (id == 0) {
            comm1 = secs_comm*batches_avg;
            ba = ((1 - overhead) * comm1) / (overhead * secs_train);
            new_ba = round(ba);
            if (new_ba < max_ba)
                batches_avg = std::max(1, new_ba);
            else 
                batches_avg = max_ba;
            printf("[DISTR] method LIMIT OVERHEAD %2.1f%%, batches_avg %d -->  %d \n", overhead * 100.0, prev_ba, batches_avg);
        }
#ifdef cMPI
        MPICHECK(MPI_Bcast(&batches_avg, 1, MPI_INT, 0, MPI_COMM_WORLD));
#endif  
    } else
        printf("[DISTR] method LIMIT OVERHEAD is not selected. batches_avg unchanged\n"); 
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
   // int id = get_id_distributed();
    float losses = net->get_losses()[index];
    bool result;

    if (id == 0)
        if (epoch > patience) {
                printf("[DISTR] prev_loss: %f, loss: %f\n", prev_losses, losses);
            if (losses > (delta+prev_losses)) { // More losses than before
                printf("[DISTR] Early Stopping! (loss %f  > delta %f + prev_losses %f)\n", losses, delta, prev_losses);
                result = true;
            } else {  // OK
                result = false;
            }
            if (losses < prev_losses) {  // new optmimal value
                printf("[DISTR] new Best loss (prev_loss: %f, loss: %f)\n", prev_losses, losses);
                prev_losses = losses;            
                result = false;
            }
        } else 
            result = false;
    
    if (is_mpi_distributed()) {
#ifdef cMPI       
        MPICHECK(MPI_Bcast(&result, 1, MPI_BYTE, 0, MPI_COMM_WORLD));
#endif    
    }
    return result;
}

bool early_stopping_on_metric_var(Net* net, int index, float delta, int patience, int epoch) {
    //int id = get_id_distributed();
    float metrics = net->get_metrics()[index];
    bool result;

    if (id == 0)
        if (epoch > patience) {
            if (metrics > prev_metrics) {// OK
                if ((metrics - prev_metrics) < delta) {
                    printf("[DISTR] Early Stopping! ((metric %f-prev_metric %f) < delta %f)\n", metrics, prev_metrics, delta);
                    result = true;
                } else { // New optimal value
                    prev_metrics = metrics;
                    result = false;
                }
            } else if ((prev_metrics - metrics) > delta) {  // Worse results
                    printf("[DISTR] Early Stopping! ((prev_metric %f-metric %f) > delta %f)\n", prev_metrics, metrics, delta);
                    result = true;
                } else {
                    result = false;
                }
        } else
            result = false;

    if (is_mpi_distributed()) {
#ifdef cMPI
        MPICHECK(MPI_Bcast(&result, 1, MPI_BYTE, 0, MPI_COMM_WORLD));
#endif    
    }
    return result;
}

bool early_stopping_on_metric(Net* net, int index, float goal, int patience, int epoch) {
    //int id=get_id_distributed();
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
    
    if (is_mpi_distributed()) {
#ifdef cMPI
        MPICHECK(MPI_Bcast(&result, 1, MPI_BYTE, 0, MPI_COMM_WORLD));
#endif
    }
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

void quantize(Tensor *B, int n_int, int n_frac){
// Debug!
    int n=n_int+n_frac;
    int max_int = std::pow(2,n);
    int scaling = std::pow(2,n_frac);
    float max_val=std::pow(2,n_int)-std::pow(2,-n_frac);
        B->clamp_(-max_val,max_val);
        B->mult_(scaling);
        B->round_();
        B->div_(scaling);

    return;
}

void CPU_quantize_network_distributed(Net* net, int nbits_int, int nbits_frac) {
    float * myptr;
    int count;
    // int n_procs;
    printf("[DISTR] %s bits int %d, bits_frac %d\n", __func__, nbits_int, nbits_frac);
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

void GPU_quantize_network_distributed(Net* net, int nbits_int, int nbits_frac) {
    // int n_procs;
    printf("[DISTR] %s bits int %d, bits_frac %d\n", __func__, nbits_int, nbits_frac);
    for (int i = 0; i < net->layers.size(); i++) {
        if (net->layers[i]->trainable) {
            for (int j = 0; j < net->layers[i]->get_trainable_params_count(); j++) {
                //printf("\n===== Proc %d Batch %d Bucle ii=%d jj=%d size=%d\n", id, j, ii,jj,count );
                quantize(net->snets[0]->layers[i]->params[j], nbits_int, nbits_frac);

            }
        }
    }
}

void gen_unique_random_list(int* vektor, int n) {

    int in, im;
    int tmp;

    im = 0;

    for (in = 0; in < n; ++in)
        vektor[in] = in;

    for (in = 0; in < n; ++in) {
        im = rand() % n;
        if (in != im) {
            tmp = vektor[in];
            vektor[in] = vektor[im];
            vektor[im] = tmp;
        }
    }

}

void gen_unique_random_list_LFSR(int* vektor, int n) {

    int rnd;
    int i, j;

    vektor[0] = rand() % n;

    for (i = 1; i < n; i++) {
        rnd = rand() % n;

        for (j = 0; j < i; j++) {
            if (rnd == vektor[j]) {
                i--;
                break;
            }
        }

        if (j >= i)
            vektor[i] = rnd;
    }
}


void loadXY(int buffer_index, int ds_ptr) {
      
        unsigned char bytesX[n_sizeX];
        unsigned char bytesY[n_sizeY];
	int i,j,index;
        int err;
        long int pos;
        off_t posX, posY;
       
        size_t n_read;
        // Random batches of sequential items

    //printf("%s ds_ptr=%ld", __func__, ds_ptr);
   

    pos = rand() % dg_num_batches;
    //pos=list[0];
    //pos=0;

#ifdef DEBUG_DONE
    done_batches[pos] += 1;
#endif 
    posX = (off_t) (ndimX + 1) * sizeof (int)+(off_t) pos * n_sizeX * sizeof (unsigned char);
    //printf("%s ds_ptr=%d pos=%ld  \n", __func__, ds_ptr, list[ds_ptr]);
    err = fseeko(fpX, posX, SEEK_SET);
    if (err) {
        msg("Error fseek ", __func__);
    }
    
    n_read = fread(bytesX, sizeof (unsigned char), n_sizeX, fpX);
    if (n_read != n_sizeX) {
        msg("Error fread ", __func__);
    }

    //printf("%s count=%d buffer_index=%d ptr_out=%d pos=%ld\n",__func__,buffer_count,buffer_index,ptr_out,pos);
    if (DEBUG) printf("LOAD:");
    for (i = 0; i < dg_batch_size; i++) {
        for (j = 0; j < shape_sizeX; j++) {
            index = buffer_index * n_sizeX + i * shape_sizeX + j;
            bufferX[index] = (float) bytesX[i * shape_sizeX + j];
            if (DEBUG) printf("[%d %3.1f ] ", index, bufferX[index]);
        }
    }
    if (DEBUG) printf("\n");
    posY = (off_t) (ndimY + 1) * sizeof (int)+(off_t) pos * n_sizeY * sizeof (unsigned char);

    //printf("%s pos=%ld \n", __func__, pos);
    err = fseeko(fpY, posY, SEEK_SET);
    if (err)
        msg("Error fseek ", __func__);
    n_read = fread(bytesY, sizeof (unsigned char), n_sizeY, fpY);
    if (n_read != n_sizeY) {
        msg("Error fread ", __func__);
    }
    if (DEBUG) printf("LOAD:");
    for (i = 0; i < dg_batch_size; i++) {
        for (j = 0; j < shape_sizeY; j++) {
            index = buffer_index * n_sizeY + i * shape_sizeY + j;
            bufferY[index] = (float) bytesY[i * shape_sizeY + j];
            if (DEBUG) printf("[%d %3.1f ] ", index, bufferY[index]);
        }
    }
    if (DEBUG) printf("\n");

}

void loadXY_perfect(int buffer_index, int ds_ptr, bool perfect) {
   
//    unsigned char bytesX[n_sizeX];
//    unsigned char bytesY[n_sizeY];
    unsigned char* bytesX;
    unsigned char* bytesY;
    long i, j;
    long index;
    int err;
    long int pos;
    off_t posX, posY;
    size_t n_read;
    // Random batches of sequential items
    bytesX=(unsigned char*) malloc(n_sizeX);
    bytesY=(unsigned char*) malloc(n_sizeY);
    
    //printf("%s ds_ptr=%ld", __func__, ds_ptr);
    if (perfect)
        pos = list[ds_ptr];
    else {
        pos = rand() % dg_num_batches;
    }
    //fprintf(tmp_fp,"%s sizeof(size_t)=%d perfect=%d buffer_index=%d num_batches=%d sizes= %ld %ld pos=%d\n", __func__, sizeof(size_t), perfect, buffer_index, dg_num_batches, n_sizeX, n_sizeY,pos);
    //fflush(tmp_fp);
    //pos=0;

#ifdef DEBUG_DONE
    done_batches[pos] += 1;
#endif 
    posX = (off_t) (ndimX + 1) * sizeof (int)+(off_t) pos * n_sizeX * sizeof (unsigned char);
    //fprintf(tmp_fp,"%s ds_ptr=%d pos=%ld  \n", __func__, ds_ptr, posX);
    //fflush(tmp_fp);
    err = fseeko(fpX, posX, SEEK_SET);
    if (err) {
              msg("Error fseek ", __func__);
    }

    n_read = fread(bytesX, sizeof (unsigned char), n_sizeX, fpX);
    if (n_read != n_sizeX) {
        printf("%s n_read %d n_size %ld\n", __func__, n_read, n_sizeX);
              msg("Error freadX ", __func__);
    }

     

    //printf("%s count=%d buffer_index=%d ptr_out=%d pos=%ld\n",__func__,buffer_count,buffer_index,ptr_out,pos);
    if (DEBUG) 
        printf("LOAD:");
    #pragma omp parallel for private(j,index)  
    for (i = 0; i < dg_batch_size; i++) {
        for (j = 0; j < shape_sizeX; j++) {
            index = buffer_index * n_sizeX + i * shape_sizeX + j;
            bufferX[index] = (float) bytesX[i * shape_sizeX + j];
            //if (DEBUG)
               // printf("[%d %3.1f ] ", index, bufferX[index]);
        }
    }
    if (DEBUG) printf("\n");
    posY = (off_t) (ndimY + 1) * sizeof (int)+(off_t) pos * n_sizeY * sizeof (unsigned char);

 
    //printf("%s pos=%ld \n", __func__, pos);
    err = fseeko(fpY, posY, SEEK_SET);
    if (err)
        msg("Error fseek ", __func__);
    n_read = fread(bytesY, sizeof (unsigned char), n_sizeY, fpY);
    if (n_read != n_sizeY) {
        printf("%s n_read %d n_size %ld\n", __func__, n_read, n_sizeX);
        msg("Error freadY ", __func__);
    }
    if (DEBUG) printf("LOAD:");
    #pragma omp parallel for private(j,index) 
    for (i = 0; i < dg_batch_size; i++) {
        for (j = 0; j < shape_sizeY; j++) {
            index = buffer_index * n_sizeY + i * shape_sizeY + j;
            bufferY[index] = (float) bytesY[i * shape_sizeY + j];
//            bufferY[index] = (float) 0;
            //if (DEBUG) 
              //  printf("[%d %3.1f ] ", index, bufferY[index]);
        }
    }
    if (DEBUG) printf("\n");
    free(bytesX);
    free(bytesY);
    
}

void loadXY_Rand(int buffer_index) {
    unsigned char bytesX[shape_sizeX];
    unsigned char bytesY[shape_sizeY];
    int i, j, index;
    long pos, posX, posY;
    // Random batches of randomly selected items

    
    if (DEBUG) printf("LOAD:");
    for (i = 0; i < dg_batch_size; i++) {
        pos = rand() % dataset_size;
        done_images[pos]+=1;
        posX = (ndimX + 1) * sizeof (int)+(long) pos * shape_sizeX * sizeof (unsigned char);
        fseek(fpX, posX, SEEK_SET);
        fread(&bytesX,sizeof (unsigned char), shape_sizeX,  fpX);
        for (j = 0; j < shape_sizeX; j++) {
            index = buffer_index * dg_batch_size * shape_sizeX + i * shape_sizeX + j;
            bufferX[index] = (float) bytesX[j];
            if (DEBUG) printf("[%d %3.1f ] ", index, bufferX[index]);
        }
        posY = (ndimY + 1) * sizeof (int)+(long) pos * shape_sizeY * sizeof (unsigned char);
        fseek(fpY, posY, SEEK_SET);
        fread(&bytesY,sizeof (unsigned char), shape_sizeY,  fpY);
        for (j = 0; j < shape_sizeY; j++) {
            index = buffer_index * dg_batch_size * shape_sizeY + i * shape_sizeY + j;
            bufferY[index] = (float) bytesY[j];
            if (DEBUG) printf("[%d %3.1f ] ", index, bufferY[index]);
        }
    }
}

void load(FILE* fp, int ndim, int shape, long record, float* buffer, int buffer_index) {
	unsigned char bytes[shape*dg_batch_size];
	int i,j,index;
    long pos;
        // Batches con items secuenciales y alineados en multiplo de BS
   

        //printf("%s ndim=%d shape=%d batch_size=%d \n",__func__,ndim,shape,batch_size);
 	
        //printf("%s record=%ld \n", __func__, record);
	pos = (ndim+1)*sizeof(int)+(long) record*shape*dg_batch_size*sizeof(unsigned char);
  //printf("%s pos=%ld \n", __func__, pos);
	fseek(fp,pos, SEEK_SET);
	fread(&bytes,sizeof(unsigned char), shape*dg_batch_size,  fp);
	//printf("%s count=%d buffer_index=%d ptr_out=%d pos=%ld\n",__func__,buffer_count,buffer_index,ptr_out,pos);
	if (DEBUG) printf("LOAD:");
	for (i=0; i<dg_batch_size; i++) {
		for (j=0; j<shape; j++) {
		index = buffer_index*dg_batch_size*shape+i*shape+j;	
		buffer[index]=(float) bytes[i*shape+j];	
		if (DEBUG) printf("[%d %3.1f ] ",index, buffer[index]);
		}
	}	
	if (DEBUG) printf("\n");
//	sem_post(&imprimir);
}



void copy_from_buffer(float* buffer, int buffer_index, int shape, float* batch) {
    int i, j;
    int index;

    //	sem_wait(&imprimir);
    if (DEBUG) printf("COMSUMER:");
    index=buffer_index * dg_batch_size * shape;
    memcpy(batch, &buffer[index], dg_batch_size*shape*sizeof(float));
    /*
    for (i = 0; i < dg_batch_size; i++) {
        for (j = 0; j < shape; j++) {
            index = buffer_index * dg_batch_size * shape + i * shape + j;
            batch[i * shape + j] = buffer[index];
            if (DEBUG) printf("[%d %3.1f ] ",index, buffer[index]);
        }
    }
    if (DEBUG) printf("\n");
     */
}

__thread int curr_ptr = 0;
__thread int curr_ds_ptr = 0;

void * producer(void* arg) {
    bool run_producer = true;
    //__thread long record;
    while (run_producer) {
        sem_wait(&llenar);
        sem_wait(&dmutex);
        curr_ptr = ptr_in;
        //curr_ds_ptr = ds_ptr;
        //ds_ptr++;
        //if (ds_ptr > dg_num_batches)
        //   run_producer = false;
        if (DEBUG)
            // printf("%s ptr_in %d count= %d\n", __func__, ptr_in, buffer_count);
            ptr_in++;
        if (ptr_in >= dg_buffer_size) ptr_in = 0;
        sem_post(&dmutex);
        //if (run_producer) {
        //loadXY(curr_ptr, curr_ds_ptr);
        loadXY_perfect(curr_ptr, curr_ds_ptr, false);
        //loadXY_Rand(curr_ptr);
        //record=rand() % dg_num_batches;
        //load(fpX, ndimX, shape_sizeX, record, bufferX, curr_ptr);
        //load(fpY, ndimY, shape_sizeY, record, bufferY, curr_ptr);
        sem_wait(&dmutex);
        buffer_count++;
        sem_post(&dmutex);
        sem_post(&vaciar);
        //}
    }
}

void * producer_perfect(void* arg) {
    bool run_producer = true;
    pid_t tid = pthread_self();
    
    //__thread long record;
    while (run_producer) {
        sem_wait(&llenar);
        sem_wait(&dmutex);
        curr_ptr = ptr_in;
        curr_ds_ptr = ds_ptr;
        ds_ptr++;
        if (ds_ptr > dg_num_batches)
            run_producer = false;
        //if (DEBUG)
        //fprintf(tmp_fp,"%s ptr_in %d count= %d\n", __func__, ptr_in, buffer_count);
        //fflush(tmp_fp);
        ptr_in++;
        if (ptr_in >= dg_buffer_size) ptr_in = 0;
        sem_post(&dmutex);
        if (run_producer) {
            
            loadXY_perfect(curr_ptr, curr_ds_ptr, dg_perfect);
            //loadXY_Rand(curr_ptr);
            //record=rand() % dg_num_batches;
            //load(fpX, ndimX, shape_sizeX, record, bufferX, curr_ptr);
            //load(fpY, ndimY, shape_sizeY, record, bufferY, curr_ptr);
            sem_wait(&dmutex);
            buffer_count++;
            sem_post(&dmutex);
            sem_post(&vaciar);
        }
        if ((ds_ptr % 10) == 0) {
            fprintf(tmp_fp, "Thread: %ld ptr_in=%d ptr_out=%d ds_ptr=%d buffer_count=%d\n", tid, ptr_in, ptr_out, ds_ptr, buffer_count);
            fflush(tmp_fp);
        }
    }
}

void* get_batch(Tensor* in, Tensor* out) {

    //printf("%s running\n", __func__);
    sem_wait(&vaciar);

    //printf("%s ptr_out %d count= %d\n", __func__, ptr_out, buffer_count);

    
    copy_from_buffer(bufferX, ptr_out, shape_sizeX, in->ptr);
    copy_from_buffer(bufferY, ptr_out, shape_sizeY, out->ptr);
  
    //printf("COPY FROM BUFFER");
    /*
    for (int i = 0; i < dg_batch_size; i++) {
        for (int j = 0; j < shape_sizeY; j++) {
            if (DEBUG) printf("[%d %3.1f ] ",index, (float) out->ptr[i * shape_sizeY + j]);
        }
    }
    */

    ptr_out++;
    if (ptr_out >= dg_buffer_size) ptr_out = 0;


    sem_wait(&dmutex);
    buffer_count--;
    sem_post(&dmutex);
    sem_post(&llenar);
    
}

void prepare_data_generator(const string &filenameX, const string &filenameY, int bs, int* num_batches,  bool perfect, int num_threads, int buffer_size) {
    int check_ds_size = 0;
    size_t memsizeX =0;
    size_t memsizeY =0;
    
    
    dg_perfect=perfect;
    dg_batch_size = bs;
    
    if (buffer_size>MAX_BUFFER)
        msg("Error buffer_size is too high ",__func__); 
    dg_buffer_size = buffer_size;
    
    if (num_threads>MAX_DG_THREADS)
        msg("Error num threads is too high", __func__); 
    dg_num_threads = num_threads;
    
    printf("[DISTR] %s. perfect=%d buffer_size=%d. num_threads=%d. bs=%d\n",__func__, dg_perfect, dg_buffer_size, dg_num_threads, dg_batch_size );
    
    fpX = fopen(filenameX.c_str(), "r");
    if (fpX == NULL)
        msg("Error opening X file", __func__); 
    fread(&ndimX, sizeof (int),1,  fpX);
    //printf("%s %d\n", __func__, ndimX);
    vector<int> r_shapeX(ndimX);

    fread(r_shapeX.data(), sizeof (int), ndimX, fpX);
    dataset_size = r_shapeX[0];
    //printf("%s %d\n", __func__, dataset_size);
    shape_sizeX = 1;
    for (int i = 1; i < ndimX; i++) {
        shape_sizeX *= r_shapeX[i];
    }
    n_sizeX = shape_sizeX*dg_batch_size;
    
    dg_num_batches = dataset_size/dg_batch_size;
    *num_batches = dg_num_batches;
    
    printf("[DISTR] %s. filenameX: %s shape_sizeX=%d dataset_size=%d num_batches=%d\n",__func__, filenameX.c_str(), shape_sizeX, dataset_size, dg_num_batches );
    
    fpY = fopen(filenameY.c_str(), "r");
    if (fpY == NULL)
        msg("Error opening Y file", __func__); 
    fread(&ndimY, sizeof (int), 1,  fpY);
    vector<int> r_shapeY(ndimY);
    fread(r_shapeY.data(), sizeof (int), ndimY,  fpY);
    check_ds_size = r_shapeY[0];
    shape_sizeY = 1;
    for (int i = 1; i < ndimY; i++) {
        shape_sizeY *= r_shapeY[i];
    }
    n_sizeY = shape_sizeY*dg_batch_size;
    
    printf("[DISTR] %s. filenameY: %s shape_sizeY=%d dataset_size=%d\n",__func__, filenameY.c_str(), shape_sizeY, check_ds_size );
    
    memsizeX = dg_buffer_size * n_sizeX * sizeof (float);
    bufferX = (float*) malloc(memsizeX);
     if (bufferX==NULL)
        msg("Error in malloc (bufferX)", __func__); // Exits
    //printf("%s memsize bufferX= %d\n", __func__, mem_size);
    memsizeY = dg_buffer_size * n_sizeY * sizeof (float);
    bufferY = (float*) malloc(memsizeY);
    if (bufferY==NULL)
        msg("Error in malloc (bufferY)", __func__); // Exits
    //printf("%s memsize bufferY= %d\n", __func__, mem_size);
    // Force bs length in shape:
    //r_shapeX[0] = bs;
    //r_shapeY[0] = bs;
    
    if (dataset_size != check_ds_size)
        msg("Error dataset sizes X and Y are different", __func__); // Exits
 
    

    if (dg_perfect) {
        list = (int*) malloc(dg_num_batches * sizeof (int));
        if (list == NULL)
            msg("Error in malloc (list)", __func__); // Exits
    }
    
     printf("[DISTR] %s. buffer requirements: bufferX= %.1fMB bufferY= %.1fMB list= %.1fMB \n", __func__,(float) memsizeX/(1024*1024), (float) memsizeY/(1024*1024), ((float) dg_num_batches * sizeof (int))/(1024*1024));
    
#ifdef DEBUG_DONE
    done_batches = (int*) malloc(dg_num_batches * sizeof (int));
    done_images = (int*) malloc(dataset_size * sizeof (int));
#endif
    srand(id*time(NULL));
     //printf("[DISTR] %s OK\n", __func__);
    /*
    bytesX=(unsigned char*)malloc(shape_sizeX*dg_batch_size*sizeof(unsigned char));
    if (bytesX==NULL)
        msg("Error in malloc (bytesX)", __func__); // Exits
    bytesY=(unsigned char*)malloc(shape_sizeY*dg_batch_size*sizeof(unsigned char));
     if (bytesY==NULL)
        msg("Error in malloc (bytesY)", __func__); // Exits
     */
    tmp_fp= fopen(tmp_name, "w");
    if (tmp_fp == NULL)
        msg("Error opening X file", __func__); 
   	

}

void start_data_generator(){

    int err; 
    
    
        buffer_count=0;
        ptr_in=0;
        ptr_out=0;
        ds_ptr=0;
        if (dg_perfect) {
            gen_unique_random_list(list, dg_num_batches);
        }
#ifdef DEBUG_DONE
        for (int i = 0; i < dg_num_batches; i++)
            done_batches[i] = 0;
        for (int i = 0; i < dataset_size; i++)
            done_images[i] = 0;
#endif

        if (sem_init(&dmutex, 0, 1) != 0) exit(EXIT_FAILURE);
        if (sem_init(&llenar, 0, dg_buffer_size) != 0) exit(EXIT_FAILURE);
        if (sem_init(&vaciar, 0, 0) != 0) exit(EXIT_FAILURE);
        if (sem_init(&imprimir, 0, 1) != 0) exit(EXIT_FAILURE);

        printf("[DISTR] %s creating %d thread(s): ", __func__, dg_num_threads);
        for (int i = 0; i < dg_num_threads; i++) {
            printf("%d ", i);

            err = pthread_create(&t[i], NULL, &producer_perfect, NULL);

            if (err) msg("Error creating thread", __func__); // Exits;

        }
        printf("\n");
    
}

void stop_data_generator() {
    
    for (int i = 0; i < dg_num_threads; i++)
        pthread_cancel(t[i]);
    for (int i = 0; i < dg_num_threads; i++)
        pthread_join(t[i], NULL);

    sem_destroy(&dmutex);
    sem_destroy(&llenar);
    sem_destroy(&vaciar);
    sem_destroy(&imprimir);
   
   /*
     #ifdef DEBUG_DONE
    for (int i = 0; i < dg_num_batches; i++)
        printf("done_batches[%d]=%d ", i, done_batches[i]);
    printf("\n");
     #endif
    */
   
    printf("[DISTR] %s count= %d\n", __func__, buffer_count);
}

void end_data_generator() {
   
    free(bufferX);
    free(bufferY); 
     free(list);
  //   free(bytesX);
  //     free(bytesY);
     fclose(fpX);
     fclose(fpY);
     fclose(tmp_fp);
}

int get_buffer_count() {
    return buffer_count;
}


