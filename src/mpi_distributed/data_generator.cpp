/*
 * MPI support for EDDL Library - European Distributed Deep Learning Library.
 * Version: 
 * copyright (c) 2021, Universidad Politécnica de Valencia (UPV), GAP research group
 * Date: July 2021
 * Author: GAP Research Group (UPV), contact: plopez@disca.upv.es
 * All rights reserved
 */

#define _FILE_OFFSET_BITS 64
//#define _GNU_SOURCE

#include "eddl/mpi_distributed/data_generator.h"
#include "eddl/mpi_distributed/mpi_distributed.h"

#include <chrono>

#include <sys/types.h>

#include <unistd.h>

using namespace std::chrono;



static int DEBUG = 0;


bool dg_created=false;
bool dg_running=false;

FILE* dg_fpX;
FILE* dg_fpY;
#ifdef cMPI
MPI_File dg_mfpX;
MPI_File dg_mfpY;
#endif

int dg_buffer_count = 0;
int dg_ptr_in = 0;
int dg_ptr_out = 0;
int dg_ds_ptr = 0;
sem_t dmutex;
sem_t llenar;
sem_t vaciar;
sem_t imprimir;
pthread_t t[MAX_DG_THREADS];

int dg_dataset_size;
int dg_ndimX;
int dg_ndimY;
int dg_shape_sizeX;
int dg_shape_sizeY;

int dg_batch_size = 0;
int dg_nbpp;
bool dg_distr_ds=false;
int dg_buffer_size = 0;
int dg_num_threads;
Tensor* dg_bufferX[MAX_BUFFER];
Tensor* dg_bufferY[MAX_BUFFER];
int* dg_list;
int dg_method;
char dg_tmp_name[128] = "/tmp/eddl_dg.txt";
FILE* dg_tmp_fp;
size_t dg_n_sizeX;
size_t dg_n_sizeY;


int total_dg = 0;
int nr_dg_running=0;

// For debugging purposes
//#define DEBUG_DONE 
//int done_batches[10000];
//int done_images[100000];


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


void gen_unique_random_vector(vector<int> &vektor, int n) {

    int in, im;
    int tmp;

    im = 0;

    vektor.resize(0);
    for (in = 0; in < n; ++in)
        vektor.push_back(in);

    for (in = 0; in < n; ++in) {
        im = rand() % n;
        if (in != im) {
            tmp = vektor[in];
            vektor[in] = vektor[im];
            vektor[im] = tmp;
        }
    }
    printf ("%s size %ld\n", __func__,vektor.size());
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

void loadXY(int buffer_index, int ds_ptr, int method) {
    unsigned char* bytesX;
    unsigned char* bytesY;
    long i, j;
    long index;
    int err;
    long int pos;
    off_t posX, posY;
    int n_read;
#ifdef cMPI
    //MPI_Offset pos;
    MPI_Status status;
#endif
   
    bytesX = (unsigned char*) malloc(dg_n_sizeX);
    if (bytesX == NULL)
        msg("Error bytesX memory allocation", __func__);
    bytesY = (unsigned char*) malloc(dg_n_sizeY);
    if (bytesY == NULL)
        msg("Error bytesY memory allocation", __func__);

    //printf("%s ds_ptr=%ld", __func__, ds_ptr);
    //  printf("1 %s ds_ptr=%ld\n", __func__, ds_ptr);
    if (method == DG_PERFECT) {
        pos = dg_list[ds_ptr];
    } else if (method == DG_RANDOM) {
        pos = rand() % dg_nbpp;
    } else if (method == DG_LIN) {
        pos = ds_ptr;
    } else {
        msg("Error unknown method", __func__);
    }

    //printf("%s pos=%d\n", __func__, pos);
    //fprintf(tmp_fp,"%s sizeof(size_t)=%d perfect=%d buffer_index=%d num_batches=%d sizes= %ld %ld pos=%d\n", __func__, sizeof(size_t), perfect, buffer_index, dg_nbpp, n_sizeX, n_sizeY,pos);
    //fflush(tmp_fp);
    //pos=0;
  //  printf ("%s id=%d method=%d bs=%d pos=%d ds_ptr=%d\n", __func__, get_id_distributed(),method, dg_batch_size, pos, ds_ptr);

#ifdef DEBUG_DONE
    done_batches[pos] += 1;
    for (int i=0; i<dg_batch_size; i++)
        done_images[pos*dg_batch_size+i]+=1;
#endif 
    posX = (off_t) (dg_ndimX + 1) * sizeof (int)+(off_t) pos * dg_n_sizeX * sizeof (unsigned char);
    //    fprintf(tmp_fp,"%s ds_ptr=%d pos=%ld  \n", __func__, ds_ptr, posX);
    //    fflush(tmp_fp);
    if (is_mpi_distributed() == 0) {
        err = fseeko(dg_fpX, posX, SEEK_SET);
        if (err) {
            msg("Error fseek ", __func__);
        }
        n_read = fread(bytesX, sizeof (unsigned char), dg_n_sizeX, dg_fpX);
    } else {
#ifdef cMPI
    MPICHECK(MPI_File_seek(dg_mfpX, posX, MPI_SEEK_SET));
    MPICHECK(MPI_File_read(dg_mfpX, bytesX, dg_n_sizeX, MPI_BYTE, &status));
    MPICHECK(MPI_Get_count(&status, MPI_BYTE, &n_read));
#endif
    }
    
    if (n_read != dg_n_sizeX) {
        printf("%s n_read %d n_size %d\n", __func__, n_read, dg_n_sizeX);
        msg("Error freadX ", __func__);
    }

    //printf("%s count=%d buffer_index=%d ptr_out=%d pos=%ld\n",__func__,buffer_count,buffer_index,ptr_out,pos);
    #pragma omp parallel for private(j,index)  
    for (i = 0; i < dg_batch_size; i++) {
        for (j = 0; j < dg_shape_sizeX; j++) {
            index = i * dg_shape_sizeX + j;
            //printf("%s buffer_index=%d index=%d\n",__func__,buffer_index,index);
            dg_bufferX[buffer_index]->ptr[index] = (float) bytesX[index];
            //if (DEBUG)
            // printf("[%d %3.1f ] ", index, bufferX[index]);
        }
    }
    if (DEBUG) printf("\n");
    posY = (off_t) (dg_ndimY + 1) * sizeof (int)+(off_t) pos * dg_n_sizeY * sizeof (unsigned char);

     if (is_mpi_distributed() == 0) {
        err = fseeko(dg_fpY, posY, SEEK_SET);
        if (err)
            msg("Error fseek ", __func__);
        n_read = fread(bytesY, sizeof (unsigned char), dg_n_sizeY, dg_fpY);
    } else {
#ifdef cMPI
    MPICHECK(MPI_File_seek(dg_mfpY, posY, MPI_SEEK_SET));
    MPICHECK(MPI_File_read(dg_mfpY, bytesY, dg_n_sizeY, MPI_BYTE, &status));
    MPICHECK(MPI_Get_count(&status, MPI_BYTE, &n_read));
#endif
    }
    if (n_read != dg_n_sizeY) {
        printf("%s n_read %d n_size %d\n", __func__, n_read, dg_n_sizeY);
        msg("Error freadY ", __func__);
    }

    #pragma omp parallel for private(j,index) 
    for (i = 0; i < dg_batch_size; i++) {
        for (j = 0; j < dg_shape_sizeY; j++) {
            index = i * dg_shape_sizeY + j;
            dg_bufferY[buffer_index]->ptr[index] = (float) bytesY[index];
            //            bufferY[index] = (float) 0;
            //if (DEBUG) 
            //  printf("[%d %3.1f ] ", index, bufferY[index]);
        }
    }
    if (DEBUG) printf("\n");
    free(bytesX);
    free(bytesY);
}

// Deprecated
void loadXY_perfect(int buffer_index, int ds_ptr, bool perfect) {
    unsigned char* bytesX;
    unsigned char* bytesY;
    long i, j;
    long index;
    int err;
    long int pos;
    off_t posX, posY;
    size_t n_read;

    // Random batches of sequential items
    bytesX = (unsigned char*) malloc(dg_n_sizeX);
    if (bytesX == NULL)
        msg("Error bytesX memory allocation", __func__);
    bytesY = (unsigned char*) malloc(dg_n_sizeY);
    if (bytesY == NULL)
        msg("Error bytesY memory allocation", __func__);


    //  printf("%s ds_ptr=%ld", __func__, ds_ptr);
    if (perfect)
        pos = dg_list[ds_ptr];
    else {
        pos = rand() % dg_nbpp;
    }
    // fprintf(tmp_fp,"%s sizeof(size_t)=%d perfect=%d buffer_index=%d num_batches=%d sizes= %ld %ld pos=%d\n", __func__, sizeof(size_t), perfect, buffer_index, dg_num_batches, n_sizeX, n_sizeY,pos);
    //fflush(tmp_fp);
    //pos=0;

#ifdef DEBUG_DONE
    done_batches[pos] += 1;
#endif 
    posX = (off_t) (dg_ndimX + 1) * sizeof (int)+(off_t) pos * dg_batch_size* dg_n_sizeX * sizeof (unsigned char);
    //fprintf(tmp_fp,"%s ds_ptr=%d pos=%ld  \n", __func__, ds_ptr, posX);
    //fflush(tmp_fp);
    err = fseeko(dg_fpX, posX, SEEK_SET);
    if (err) {
        msg("Error fseek ", __func__);
    }

    n_read = fread(bytesX, sizeof (unsigned char), dg_n_sizeX, dg_fpX);
    if (n_read != dg_n_sizeX) {
        printf("%s n_read %d n_size %d\n", __func__, n_read, dg_n_sizeX);
        msg("Error freadX ", __func__);
    }
    //printf("%s count=%d buffer_index=%d ptr_out=%d pos=%ld\n",__func__,buffer_count,buffer_index,ptr_out,pos);
    if (DEBUG)
        printf("LOAD:");
    #pragma omp parallel for private(j,index)  
    for (i = 0; i < dg_batch_size; i++) {
        for (j = 0; j < dg_shape_sizeX; j++) {
            index = i * dg_shape_sizeX + j;
            dg_bufferX[buffer_index]->ptr[index] = (float) bytesX[index];
            //if (DEBUG)
            // printf("[%d %3.1f ] ", index, bufferX[index]);
        }
    }
    if (DEBUG) printf("\n");
    posY = (off_t) (dg_ndimY + 1) * sizeof (int)+(off_t) pos * dg_n_sizeY * sizeof (unsigned char);


    //printf("%s pos=%ld \n", __func__, pos);
    err = fseeko(dg_fpY, posY, SEEK_SET);
    if (err)
        msg("Error fseek ", __func__);
    n_read = fread(bytesY, sizeof (unsigned char), dg_n_sizeY, dg_fpY);
    if (n_read != dg_n_sizeY) {
        printf("%s n_read %d n_size %d\n", __func__, n_read, dg_n_sizeY);
        msg("Error freadY ", __func__);
    }
    if (DEBUG) printf("LOAD:");
    #pragma omp parallel for private(j,index) 
    for (i = 0; i < dg_batch_size; i++) {
        for (j = 0; j < dg_shape_sizeY; j++) {
            index = i * dg_shape_sizeY + j;
            dg_bufferY[buffer_index]->ptr[index] = (float) bytesY[index];
            //            bufferY[index] = (float) 0;
            //if (DEBUG) 
            //  printf("[%d %3.1f ] ", index, bufferY[index]);
        }
    }
    if (DEBUG) printf("\n");
    free(bytesX);
    free(bytesY);

}

/*
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
            dbufferX[index] = (float) bytesX[j];
            if (DEBUG) printf("[%d %3.1f ] ", index, bufferX[index]);
        }
        posY = (ndimY + 1) * sizeof (int)+(long) pos * shape_sizeY * sizeof (unsigned char);
        fseek(fpY, posY, SEEK_SET);
        fread(&bytesY,sizeof (unsigned char), shape_sizeY,  fpY);
        for (j = 0; j < shape_sizeY; j++) {
            index = buffer_index * dg_batch_size * shape_sizeY + i * shape_sizeY + j;
            dbufferY[index] = (float) bytesY[j];
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

}
 */


void copy_from_buffer(float* buffer, int buffer_index, int shape, float* batch) {
    int i, j;
    int index;


    if (DEBUG) printf("COMSUMER:");
    index = buffer_index * dg_batch_size * shape;
    memcpy(batch, &buffer[index], dg_batch_size * shape * sizeof (float));
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

void * producer_perfect(void* arg) {
    bool run_producer = true;
    pid_t tid = pthread_self();
    double loadsecs=0;
    int id=get_id_distributed();

    //__thread long record;
    while (run_producer) {
        sem_wait(&llenar);
        sem_wait(&dmutex);
        curr_ptr = dg_ptr_in;
        curr_ds_ptr = dg_ds_ptr;
        dg_ds_ptr++;
        if (dg_ds_ptr > dg_nbpp)
            run_producer = false;
        //if (DEBUG)
        //fprintf(tmp_fp,"%s ptr_in %d count= %d\n", __func__, ptr_in, buffer_count);
        //fflush(tmp_fp);
        dg_ptr_in++;
        if (dg_ptr_in >= dg_buffer_size) dg_ptr_in = 0;
        sem_post(&dmutex);
        if (run_producer) {
            TIME_POINT1(load);
            if (dg_distr_ds==true) {
                loadXY(curr_ptr, curr_ds_ptr, dg_method);
            } else {
                loadXY(curr_ptr, id * dg_nbpp + curr_ds_ptr, dg_method);
            }

                /*
            if (is_mpi_distributed() == 0)
                loadXY_perfect(curr_ptr, id * dg_nbpp + curr_ds_ptr, dg_perfect);
            else
                loadXY_perfect_distr(curr_ptr, id * dg_nbpp + curr_ds_ptr, dg_perfect);
                     * */
            TIME_POINT2(load, loadsecs);
        
            sem_wait(&dmutex);
            dg_buffer_count++;
            sem_post(&dmutex);
            sem_post(&vaciar);
        }
        if ((dg_ds_ptr % 1) == 0) {
            fprintf(dg_tmp_fp, "Thread: %d ptr_in=%d ptr_out=%d ds_ptr=%d avg load time=%2.4f s. buffer_count=%d\n", tid, dg_ptr_in, dg_ptr_out, id * dg_nbpp +dg_ds_ptr, (loadsecs * dg_num_threads) / dg_ds_ptr, dg_buffer_count);
            //fflush(dg_tmp_fp);
        }
    }
    return NULL;
}

void* get_batch(Tensor* in, Tensor* out) {

    if (dg_created==false)
        msg("Error DG was not created ", __func__);
    if (dg_running==false)
        msg("Error DG is not running", __func__);
     
    sem_wait(&vaciar);

#pragma omp parallel sections 
    {
#pragma omp  section 
        {
            Tensor::copy(dg_bufferX[dg_ptr_out], in);
        }
#pragma omp  section
        Tensor::copy(dg_bufferY[dg_ptr_out], out);
    }


    //in->reallocate(bufferX[ptr_out]);
    //out->reallocate(bufferY[ptr_out]);
    //in->ptr = bufferX[ptr_out]->ptr;
    //out->ptr = bufferY[ptr_out]->ptr;

    //copy_from_buffer(bufferX, ptr_out, shape_sizeX, in->ptr);
    //copy_from_buffer(bufferY, ptr_out, shape_sizeY, out->ptr);

    //printf("COPY FROM BUFFER");
    /*
    for (int i = 0; i < dg_batch_size; i++) {
        for (int j = 0; j < shape_sizeY; j++) {
            if (DEBUG) printf("[%d %3.1f ] ",index, (float) out->ptr[i * shape_sizeY + j]);
        }
    }
     */

    dg_ptr_out++;
    if (dg_ptr_out >= dg_buffer_size) dg_ptr_out = 0;


    sem_wait(&dmutex);
    dg_buffer_count--;
    sem_post(&dmutex);
    sem_post(&llenar);
    return NULL;
}

void* prepare_data_generator(int dg_id, const string &filenameX, const string &filenameY, int bs, bool distr_ds, int* dataset_size, int* nbpp, int method, int num_threads, int buffer_size) {
    int check_ds_size = 0;
    size_t memsizeX = 0;
    size_t memsizeY = 0;
     int id=get_id_distributed();
     int n_procs=get_n_procs_distributed();
     int global_batch;

    if (dg_created)
        msg("Error DG already prepared", __func__);
        
    if (dg_id >= DG_MAX)
        msg("Error number of data generators exceeded", __func__);

    dg_method = method;
    
    //set_batch_distributed(&global_batch, &dg_batch_size, bs, DIV_BATCH);
    dg_batch_size = bs;
    dg_distr_ds=distr_ds;

    if (buffer_size > MAX_BUFFER)
        msg("Error buffer_size is too high ", __func__);
    dg_buffer_size = buffer_size;

    if (num_threads > MAX_DG_THREADS)
        msg("Error num threads is too high", __func__);
    dg_num_threads = num_threads;

    if (id == 0)
        printf("[DISTR] %s. method=%d buffer_size=%d. num_threads=%d. bs=%d. Using %s\n", __func__, dg_method, dg_buffer_size, dg_num_threads, dg_batch_size, (is_mpi_distributed() == 0 ? "fopen()" : "MPI_File_open()"));

    if (is_mpi_distributed()==0) {
        dg_fpX = fopen(filenameX.c_str(), "r");
        if (dg_fpX == NULL)
            msg("Error opening X file", __func__);
        fread(&dg_ndimX, sizeof (int), 1, dg_fpX);
    } else {
#ifdef cMPI
        MPI_File_open(MPI_COMM_WORLD, filenameX.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &dg_mfpX);
        if (dg_mfpX == NULL)
            msg("Error opening X file", __func__);
        MPI_File_read_all(dg_mfpX, &dg_ndimX, 1, MPI_INT, MPI_STATUS_IGNORE);
#endif
    }
    //printf("%s %d\n", __func__, ndimX);
    vector<int> r_shapeX(dg_ndimX);

    if (is_mpi_distributed()==0)
        fread(r_shapeX.data(), sizeof (int), dg_ndimX, dg_fpX);
    else {
#ifdef cMPI
        MPI_File_read_all(dg_mfpX, r_shapeX.data(), dg_ndimX, MPI_INT, MPI_STATUS_IGNORE);
#endif
    }
    dg_dataset_size = r_shapeX[0];
    *dataset_size = r_shapeX[0];
    //printf("%s %d\n", __func__, dataset_size);
    dg_shape_sizeX = 1;
    for (int i = 1; i < dg_ndimX; i++) {
        dg_shape_sizeX *= r_shapeX[i];
    }
    // Force bs length in shape:
    r_shapeX[0] = dg_batch_size;

    dg_n_sizeX = dg_shape_sizeX*dg_batch_size;

    // Set nbpp
    dg_nbpp = set_NBPP_distributed(dg_dataset_size, dg_batch_size, dg_distr_ds);
    *nbpp = dg_nbpp;

    if (id == 0)
        printf("[DISTR] %s. filenameX: %s shape_sizeX=%d dataset_size=%d num_batches=%d\n", __func__, filenameX.c_str(), dg_shape_sizeX, dg_dataset_size, dg_nbpp);

    if (is_mpi_distributed()==0) {
        dg_fpY = fopen(filenameY.c_str(), "r");
        if (dg_fpY == NULL)
            msg("Error opening Y file", __func__);
        fread(&dg_ndimY, sizeof (int), 1, dg_fpY);
    } else {
#ifdef cMPI
        MPI_File_open(MPI_COMM_WORLD, filenameY.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &dg_mfpY);
        if (dg_mfpY == NULL)
            msg("Error opening Y file", __func__);
        MPI_File_read_all(dg_mfpY, &dg_ndimY, 1, MPI_INT, MPI_STATUS_IGNORE);
#endif
    }
    vector<int> r_shapeY(dg_ndimY);
    if (is_mpi_distributed()==0)
        fread(r_shapeY.data(), sizeof (int), dg_ndimY, dg_fpY);
    else {
#ifdef cMPI
        MPI_File_read_all(dg_mfpY, r_shapeY.data(), dg_ndimY, MPI_INT, MPI_STATUS_IGNORE);
#endif
    }
    check_ds_size = r_shapeY[0];
    dg_shape_sizeY = 1;
    for (int i = 1; i < dg_ndimY; i++) {
        dg_shape_sizeY *= r_shapeY[i];
    }
    // Force bs length in shape:
    r_shapeY[0] = dg_batch_size;
    dg_n_sizeY = dg_shape_sizeY*dg_batch_size;

    if (id == 0)
        printf("[DISTR] %s. filenameY: %s shape_sizeY=%d dataset_size=%d\n", __func__, filenameY.c_str(), dg_shape_sizeY, check_ds_size);

    for (int i = 0; i < dg_buffer_size; i++) {
        dg_bufferX[i] = new Tensor(r_shapeX);
        dg_bufferY[i] = new Tensor(r_shapeY);
    }
    memsizeX = dg_buffer_size * dg_n_sizeX * sizeof (float);
    memsizeY = dg_buffer_size * dg_n_sizeY * sizeof (float);

    if (dg_dataset_size != check_ds_size)
        msg("Error dataset sizes X and Y are different", __func__); // Exits


/*
    if (dg_method==DG_PERFECT) {
        dg_list = (int*) malloc(dg_nbpp * n_procs * sizeof (int));
        if (dg_list == NULL)
            msg("Error in malloc (list)", __func__); // Exits
    }
 * */

    if (id == 0)
        printf("[DISTR] %s. buffer requirements: bufferX= %.1fMB bufferY= %.1fMB\n", __func__, (float) memsizeX / (1024 * 1024), (float) memsizeY / (1024 * 1024));

#ifdef DEBUG_DONE

#endif
    srand(id * time(NULL));
    //printf("[DISTR] %s OK\n", __func__);
    /*
    bytesX=(unsigned char*)malloc(shape_sizeX*dg_batch_size*sizeof(unsigned char));
    if (bytesX==NULL)
        msg("Error in malloc (bytesX)", __func__); // Exits
    bytesY=(unsigned char*)malloc(shape_sizeY*dg_batch_size*sizeof(unsigned char));
     if (bytesY==NULL)
        msg("Error in malloc (bytesY)", __func__); // Exits
     */
    dg_tmp_fp = fopen(dg_tmp_name, "w");
    if (dg_tmp_fp == NULL)
        msg("Error opening tmp file", __func__);

    dg_created=true;
    return NULL;
}

void* start_data_generator() {
    int err;
    int id=get_id_distributed();
     int n_procs=get_n_procs_distributed();

    if (dg_created == false)
        msg("Error DG was not created ", __func__);
    if (dg_running)
        msg("Error DG is already running", __func__);

    dg_buffer_count = 0;
    dg_ptr_in = 0;
    dg_ptr_out = 0;
    dg_ds_ptr = 0;
    
    if (dg_method == DG_PERFECT) {     
        if (dg_distr_ds) {
            dg_list = new int[dg_nbpp];
            gen_unique_random_list(dg_list, dg_nbpp);
        } else {
            dg_list = new int[dg_nbpp * n_procs];
            if (id == 0) {
                //gen_unique_random_vector(DG->vlist, DG->nbpp * n_procs);
                // gen_unique_random_vector(dg_vector, DG->nbpp * n_procs);
                //gen_unique_random_list(DG->list, DG->nbpp * n_procs);
                gen_unique_random_list(dg_list, dg_nbpp * n_procs);
            }
#ifdef cMPI
            if (is_mpi_distributed())
                MPICHECK(MPI_Bcast(dg_list, dg_nbpp * n_procs, MPI_INT, 0, MPI_COMM_WORLD));
#endif
        }
    }

#ifdef DEBUG_DONE
    for (int i = 0; i < dg_nbpp; i++)
        done_batches[i] = 0;
    for (int i = 0; i < dg_dataset_size; i++)
        done_images[i] = 0;
#endif

    if (sem_init(&dmutex, 0, 1) != 0) exit(EXIT_FAILURE);
    if (sem_init(&llenar, 0, dg_buffer_size) != 0) exit(EXIT_FAILURE);
    if (sem_init(&vaciar, 0, 0) != 0) exit(EXIT_FAILURE);

    if (id == 0)
        printf("[DISTR] %s creating %d thread(s): ", __func__, dg_num_threads);
    for (int i = 0; i < dg_num_threads; i++) {
        if (id == 0)
            printf("%d ", i);

        err = pthread_create(&t[i], NULL, &producer_perfect, NULL);
        if (err) msg("Error creating thread", __func__); // Exits;
    }
    if (id == 0)
        printf("\n");
    dg_running=true;
    return NULL;
}

void* stop_data_generator() {
    int id=get_id_distributed();
    int n_procs=get_n_procs_distributed();
    
    if (dg_created==false)
        msg("Error DG was not created ", __func__);
    if (dg_running==false)
        msg("Error DG is not running", __func__);
    
    for (int i = 0; i < dg_num_threads; i++)
        pthread_cancel(t[i]);
    for (int i = 0; i < dg_num_threads; i++)
        pthread_join(t[i], NULL);

    sem_destroy(&dmutex);
    sem_destroy(&llenar);
    sem_destroy(&vaciar);

    
    
#ifdef DEBUG_DONE
    
    
     printf("[DG] %s, Samples processed, per proc:\n", __func__);

    int visited = 0;
    int visited_max = 0;
    int visited_min = 10000;
    for (int i = 0; i < dg_nbpp*n_procs; i++) {
        visited += done_batches[i];
        if (done_batches[i] > visited_max)
            visited_max = done_batches[i];
        if (done_batches[i] < visited_min)
            visited_min = done_batches[i];
        //printf("done_batches[%d]=%d ", i, done_batches[i]);
        //printf("\n");
    }
    printf("Proc %d. done_batches: batches=%d visited=%d min=%d max=%d \n", id, dg_nbpp, visited, visited_min, visited_max);
    visited = 0;
    visited_max = 0;
    visited_min = 1000;

    for (int i = 0; i < dg_dataset_size; i++) {
        visited += done_images[i];
        if (done_images[i] > visited_max)
            visited_max = done_images[i];
        if (done_images[i] < visited_min)
            visited_min = done_images[i];
        //printf("done_images[%d]=%d ", i, done_images[i]);
        //printf("\n");
    }
    printf("Proc %d done_images: dataset=%d visited=%d min=%d max=%d \n", id, dg_dataset_size, visited, visited_min, visited_max);

    barrier_distributed();
    
     printf("[DG] %s, Samples processed, *after* Allreduce:\n", __func__);
#ifdef cMPI
     MPI_Allreduce(MPI_IN_PLACE, &done_batches, dg_nbpp*n_procs, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
     MPI_Allreduce(MPI_IN_PLACE, &done_images, dg_dataset_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    visited = 0;
    visited_max = 0;
    visited_min = 10000;
    for (int i = 0; i < dg_nbpp*n_procs; i++) {
        visited += done_batches[i];
        if (done_batches[i] > visited_max)
            visited_max = done_batches[i];
        if (done_batches[i] < visited_min)
            visited_min = done_batches[i];
        //printf("done_batches[%d]=%d ", i, done_batches[i]);
        //printf("\n");
    }
  printf("Proc %d. done_batches: batches=%d visited=%d min=%d max=%d \n", id, dg_nbpp, visited, visited_min, visited_max);
    visited = 0;
    visited_max = 0;
    visited_min = 1000;

    for (int i = 0; i < dg_dataset_size; i++) {
        visited += done_images[i];
        if (done_images[i] > visited_max)
            visited_max = done_images[i];
        if (done_images[i] < visited_min)
            visited_min = done_images[i];
        //printf("done_images[%d]=%d ", i, done_images[i]);
        //printf("\n");
    }
    printf("Proc %d done_images: dataset=%d visited=%d min=%d max=%d \n", id, dg_dataset_size, visited, visited_min, visited_max);
     
#endif
    
    
    /*
      #ifdef DEBUG_DONE
     for (int i = 0; i < dg_num_batches; i++)
         printf("done_batches[%d]=%d ", i, done_batches[i]);
     printf("\n");
      #endif
     */

    if (id == 0)
        printf("[DISTR] %s count= %d\n", __func__, dg_buffer_count);
    dg_running=false;
    return NULL;
}

void* end_data_generator() {
    int id=get_id_distributed();

    if (dg_created == false)
        msg("Error DG was not created ", __func__);
    if (dg_running) 
        msg("Error DG is still running", __func__);
    
    for (int i = 0; i < dg_buffer_size; i++) {
        delete dg_bufferX[i];
        delete dg_bufferY[i];
    }  
     
    free(dg_list);
    //   free(bytesX);
    //     free(bytesY);
    if (is_mpi_distributed() == 0) {
        fclose(dg_fpX);
        fclose(dg_fpY);
    } else {
#ifdef cMPI
        MPICHECK(MPI_File_close(&dg_mfpX));
        MPICHECK(MPI_File_close(&dg_mfpY));
#endif
    }
    fclose(dg_tmp_fp);
    if (id == 0)
        printf("[DISTR] %s\n", __func__);
    dg_created=false;
    return NULL;
}

int get_buffer_count() {
    return dg_buffer_count;
}


// Support for several Data Generators

vector<int> dg_vector;
int* dg_lista;

void* loadXY_DataGen(DG_Data* DG, int buffer_index, int ds_ptr) {
    //    unsigned char bytesX[n_sizeX];
    //    unsigned char bytesY[n_sizeY];
    unsigned char* bytesX;
    unsigned char* bytesY;
    long i, j;
    long index;
    int err;
    long int pos;
    off_t posX, posY;
    int n_read;
#ifdef cMPI
    MPI_Status status;
#endif
    
    //printf("=== %s %d %d\n", __func__, buffer_index, ds_ptr);
    //imprime_DG(__func__,DG);
    //imprime_buffer(DG);
    
    
    // Random batches of sequential items
    bytesX = (unsigned char*) malloc(DG->n_sizeX);
    if (bytesX == NULL)
        msg("Error bytesX memory allocation", __func__);
    bytesY = (unsigned char*) malloc(DG->n_sizeY);
    if (bytesY == NULL)
        msg("Error bytesY memory allocation", __func__);


    //  printf("1 %s ds_ptr=%ld\n", __func__, ds_ptr);
    if (DG->method==DG_PERFECT)
      //  pos = DG->list[ds_ptr];
      //pos = dg_vector[ds_ptr];
      pos = dg_lista[ds_ptr];

    else if (DG->method==DG_RANDOM) {
        pos = rand() % DG->nbpp;
    }  else if (DG->method==DG_LIN) {
        pos = ds_ptr;
    }  else  {
        msg("Error unknwon method", __func__);
    }
 
    // fprintf(tmp_fp,"%s sizeof(size_t)=%d perfect=%d buffer_index=%d num_batches=%d sizes= %ld %ld pos=%d\n", __func__, sizeof(size_t), perfect, buffer_index, dg_num_batches, n_sizeX, n_sizeY,pos);
    //fflush(tmp_fp);
    //pos=0;


    posX = (off_t) (DG->ndimX + 1) * sizeof (int)+(off_t) pos * DG->n_sizeX * sizeof (unsigned char);
    //fprintf(tmp_fp,"%s ds_ptr=%d pos=%ld  \n", __func__, ds_ptr, posX);
    //fflush(tmp_fp);
    if (is_mpi_distributed() == 0) {
        /*
        err = fseeko(DG->fpX, posX, SEEK_SET);
        if (err) {
            msg("Error fseek ", __func__);
        }
        n_read = fread(bytesX, sizeof (unsigned char), DG->n_sizeX, DG->fpX);
         */
        n_read = (int) pread(fileno(DG->fpX), bytesX, DG->n_sizeX*sizeof (unsigned char), posX);
        n_read = n_read/sizeof (unsigned char);
    } else {
#ifdef cMPI
        MPICHECK(MPI_File_seek(DG->mfpX, posX, MPI_SEEK_SET));
        MPICHECK(MPI_File_read(DG->mfpX, bytesX, DG->n_sizeX, MPI_BYTE, &status));
        MPICHECK(MPI_Get_count(&status, MPI_BYTE, &n_read));
#endif
    }

    // if (n_read != DG->n_sizeX) {
    if (n_read != DG->n_sizeX) {
        printf("%s n_read %d n_size %d\n", __func__, n_read, DG->n_sizeX);
        msg("Error freadX ", __func__);
    }
    // printf("2 %s ds_ptr=%ld\n", __func__, ds_ptr);


    //printf("%s count=%d buffer_index=%d ptr_out=%d pos=%ld\n",__func__,buffer_count,buffer_index,ptr_out,pos);
    if (DEBUG)
        printf("LOAD:");
    //#pragma omp parallel for private(j,index)  
    for (i = 0; i < DG->batch_size; i++) {
        for (j = 0; j < DG->shape_sizeX; j++) {
            index = i * DG->shape_sizeX + j;
            //DG->bufferX[buffer_index]->ptr[index] = (float) bytesX[index];
            dg_bufferX[buffer_index]->ptr[index] = (float) bytesX[index];
            //if (DEBUG)
            //printf("[%d %3.1f ] ", index, DG->bufferX[buffer_index]->ptr[index]);
            //printf("[%d ] ", index);
        }
    }
    if (DEBUG) printf("\n");
    posY = (off_t) (DG->ndimY + 1) * sizeof (int)+(off_t) pos * DG->n_sizeY * sizeof (unsigned char);
    //printf("%s pos=%ld \n", __func__, pos);

    if (is_mpi_distributed() == 0) {
        /* err = fseeko(DG->fpY, posY, SEEK_SET);
        if (err)
            msg("Error fseek ", __func__);
        n_read = fread(bytesY, sizeof (unsigned char), DG->n_sizeY, DG->fpY); */
        n_read = (int) pread(fileno(DG->fpY), bytesY, DG->n_sizeY*sizeof (unsigned char), posY);         
        n_read = n_read/sizeof (unsigned char);
    } else {
#ifdef cMPI
        MPICHECK(MPI_File_seek(DG->mfpY, posY, MPI_SEEK_SET));
        MPICHECK(MPI_File_read(DG->mfpY, bytesY, DG->n_sizeY, MPI_BYTE, &status));
        MPICHECK(MPI_Get_count(&status, MPI_BYTE, &n_read));
#endif
    }
    //if (n_read != DG->n_sizeY) {
    if (n_read != DG->n_sizeY) {
        printf("%s n_read %d n_size %d\n", __func__, n_read, DG->n_sizeY);
        msg("Error freadY ", __func__);
 }

    if (DEBUG) printf("LOAD:");
    // #pragma omp parallel for private(j,index) 
    for (i = 0; i < DG->batch_size; i++) {
        for (j = 0; j < DG->shape_sizeY; j++) {
            index = i * DG->shape_sizeY + j;
            //DG->bufferY[buffer_index]->ptr[index] = (float) bytesY[index];
            dg_bufferY[buffer_index]->ptr[index] = (float) bytesY[index];
            //printf("[%d ] ", index);
            //            bufferY[index] = (float) 0;
            //if (DEBUG) 
            //  printf("[%d %3.1f ] ", index, bufferY[index]);
        }
    }
    //      printf("4 %s ds_ptr=%ld\n", __func__, ds_ptr);
    if (DEBUG) printf("\n");
    free(bytesX);
    free(bytesY);
    //    printf("5 %s ds_ptr=%ld\n", __func__, ds_ptr);
    return NULL;
}

void* loadXY_DataGen_float(DG_Data* DG, int buffer_index, int ds_ptr) {
    //    unsigned char bytesX[n_sizeX];
    //    unsigned char bytesY[n_sizeY];
    float* bytesX;
    float* bytesY;
    long i, j;
    long index;
    int err;
    long int pos;
    off_t posX, posY;
    int n_read;
#ifdef cMPI
    MPI_Status status;
#endif
    
    //printf("=== %s %d %d\n", __func__, buffer_index, ds_ptr);
    //imprime_DG(__func__,DG);
    //imprime_buffer(DG);
    
    
    // Random batches of sequential items
    bytesX = (float*) malloc(DG->n_sizeX*sizeof(float));
    if (bytesX == NULL)
        msg("Error bytesX memory allocation", __func__);
    bytesY = (float*) malloc(DG->n_sizeY*sizeof(float));
    if (bytesY == NULL)
        msg("Error bytesY memory allocation", __func__);


    //  printf("1 %s ds_ptr=%ld\n", __func__, ds_ptr);
    if (DG->method==DG_PERFECT)
      //  pos = DG->list[ds_ptr];
      //pos = dg_vector[ds_ptr];
      pos = dg_lista[ds_ptr];

    else if (DG->method==DG_RANDOM) {
        pos = rand() % DG->nbpp;
    }  else if (DG->method==DG_LIN) {
        pos = ds_ptr;
    }  else  {
        msg("Error unknwon method", __func__);
    }
 
    // fprintf(tmp_fp,"%s sizeof(size_t)=%d perfect=%d buffer_index=%d num_batches=%d sizes= %ld %ld pos=%d\n", __func__, sizeof(size_t), perfect, buffer_index, dg_num_batches, n_sizeX, n_sizeY,pos);
    //fflush(tmp_fp);
    //pos=0;


    posX = (off_t) (DG->ndimX + 1) * sizeof (int)+(off_t) pos * DG->n_sizeX * sizeof (float);
    //fprintf(tmp_fp,"%s ds_ptr=%d pos=%ld  \n", __func__, ds_ptr, posX);
    //fflush(tmp_fp);
    if (is_mpi_distributed() == 0) {
        /*
        err = fseeko(DG->fpX, posX, SEEK_SET);
        if (err) {
            msg("Error fseek ", __func__);
        }
        n_read = fread(bytesX, sizeof (float), DG->n_sizeX, DG->fpX); */
        n_read = (int) pread(fileno(DG->fpX), bytesX, DG->n_sizeX*sizeof (float), posX);
        n_read = n_read/sizeof (float);
    } else {
#ifdef cMPI
        MPICHECK(MPI_File_seek(DG->mfpX, posX, MPI_SEEK_SET));
        MPICHECK(MPI_File_read(DG->mfpX, bytesX, DG->n_sizeX, MPI_FLOAT, &status));
        MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &n_read));
#endif
    }

    //if (n_read != DG->n_sizeX) {
    if (n_read != DG->n_sizeX) {
        printf("%s n_read %d n_size %d\n", __func__, n_read, DG->n_sizeX);
        msg("Error freadX ", __func__);
    }
    // printf("2 %s ds_ptr=%ld\n", __func__, ds_ptr);


    //printf("%s count=%d buffer_index=%d ptr_out=%d pos=%ld\n",__func__,buffer_count,buffer_index,ptr_out,pos);
    if (DEBUG)
        printf("LOAD:");
    //#pragma omp parallel for private(j,index)  
    for (i = 0; i < DG->batch_size; i++) {
        for (j = 0; j < DG->shape_sizeX; j++) {
            index = i * DG->shape_sizeX + j;
            //DG->bufferX[buffer_index]->ptr[index] = (float) bytesX[index];
            dg_bufferX[buffer_index]->ptr[index] = bytesX[index];
            //if (DEBUG)
            //printf("[%d %3.1f ] ", index, DG->bufferX[buffer_index]->ptr[index]);
            //printf("[%d ] ", index);
        }
    }
    if (DEBUG) printf("\n");
    posY = (off_t) (DG->ndimY + 1) * sizeof (int)+(off_t) pos * DG->n_sizeY * sizeof (float);
    //printf("%s pos=%ld \n", __func__, pos);

    if (is_mpi_distributed() == 0) {
        /* err = fseeko(DG->fpY, posY, SEEK_SET);
        if (err)
            msg("Error fseek ", __func__);
        n_read = fread(bytesY, sizeof (float), DG->n_sizeY, DG->fpY); */
        n_read = (int) pread(fileno(DG->fpY), bytesY, DG->n_sizeY*sizeof (float), posY);
        n_read = n_read/sizeof (float);
    } else {
#ifdef cMPI
        MPICHECK(MPI_File_seek(DG->mfpY, posY, MPI_SEEK_SET));
        MPICHECK(MPI_File_read(DG->mfpY, bytesY, DG->n_sizeY, MPI_FLOAT, &status));
        MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &n_read));
#endif
    }
    //if (n_read != DG->n_sizeY) {
    if (n_read != DG->n_sizeY) {
        printf("%s n_read %d n_size %d\n", __func__, n_read, DG->n_sizeY);
        msg("Error freadY ", __func__);
  }

    if (DEBUG) printf("LOAD:");
    // #pragma omp parallel for private(j,index) 
    for (i = 0; i < DG->batch_size; i++) {
        for (j = 0; j < DG->shape_sizeY; j++) {
            index = i * DG->shape_sizeY + j;
            //DG->bufferY[buffer_index]->ptr[index] = (float) bytesY[index];
            dg_bufferY[buffer_index]->ptr[index] = bytesY[index];
            //printf("[%d ] ", index);
            //            bufferY[index] = (float) 0;
            //if (DEBUG) 
            //  printf("[%d %3.1f ] ", index, bufferY[index]);
        }
    }
    //      printf("4 %s ds_ptr=%ld\n", __func__, ds_ptr);
    if (DEBUG) printf("\n");
    free(bytesX);
    free(bytesY);
    //    printf("5 %s ds_ptr=%ld\n", __func__, ds_ptr);
    return NULL;
}

// Already defined
//__thread int curr_ptr = 0;
//__thread int curr_ds_ptr = 0;

void* producer_DataGen(void* arg) {
    bool run_producer = true;
    pid_t tid = pthread_self();
    double loadsecs=0;
    int curr_ptr=0;
    int curr_ds_ptr=0;
    int id=get_id_distributed();
    DG_Data* DG = (DG_Data*) arg;

//    printf("%s tid=%ld\n", __func__, tid);

    //imprime_DG(__func__,DG);
    //   pthread_detach(tid);

    //__thread long record;
    fprintf(DG->tmp_fp, "Producer: Init    Thread %ld ptr_in=%d ptr_out=%d ds_ptr=%d avg load time=%2.4f s. buffer_count=%d produced=%d consumed=%d\n", tid, DG->ptr_in, DG->ptr_out, DG->ds_ptr, 0.0, DG->buffer_count, DG->total_produced, DG->total_consumed);
    fflush(DG->tmp_fp);
    while (run_producer) {
        
        sem_wait(&llenar);
        sem_wait(&dmutex);
        if (DG->ds_ptr < DG->nbpp) {
			curr_ptr = DG->ptr_in;
			curr_ds_ptr = DG->ds_ptr;
			DG->ds_ptr++;
			DG->ptr_in = (DG->ptr_in+1) % DG->buffer_size;
		} else 
            run_producer = false;
        //if (DEBUG)
        //  fprintf(stdout,"1 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
        //  fflush(stdout);
        //fprintf(DG->tmp_fp,"%s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
        //fflush(DG->tmp_fp);
        //     fprintf(stdout,"2 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
        //     fflush(stdout);
        sem_post(&dmutex);
        
        if (run_producer) {

        TIME_POINT1(load);
        //       fprintf(stdout,"3 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
        //  fflush(stdout);
        if (DG->distr_ds) {
            if (DG->datatype==DG_BYTE)
                loadXY_DataGen(DG, curr_ptr, curr_ds_ptr);
            else if (DG->datatype==DG_FLOAT)
                loadXY_DataGen_float(DG, curr_ptr, curr_ds_ptr);
            else {
                msg("Error DG data type not supported ", __func__);
            }
        } else {
            
            if (DG->datatype==DG_BYTE)
                loadXY_DataGen(DG, curr_ptr, id * DG->nbpp + curr_ds_ptr);
            else if (DG->datatype==DG_FLOAT)
                loadXY_DataGen_float(DG, curr_ptr, id * DG->nbpp + curr_ds_ptr);
            else {
                msg("Error DG data type not supported ", __func__);
            }
        }
        TIME_POINT2(load, loadsecs);
        //        fprintf(stdout,"4 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
        //  fflush(stdout);

        sem_wait(&dmutex);
        DG->total_produced++;
        DG->buffer_count++;
        sem_post(&dmutex);
        sem_post(&vaciar);
        }
        //    fprintf(stdout,"5 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
        //   fflush(stdout);
        if ((DG->ds_ptr % 1) == 0) {
                fprintf(DG->tmp_fp, "Producer: Running Thread %ld ptr_in=%d ptr_out=%d ds_ptr=%d avg load time=%2.4f s. buffer_count=%d produced=%d consumed=%d\n", tid, DG->ptr_in, DG->ptr_out, DG->ds_ptr, (loadsecs*DG->num_threads)/DG->ds_ptr, DG->buffer_count, DG->total_produced, DG->total_consumed);
                fflush(DG->tmp_fp);
        }
        if (run_producer==false) {
                fprintf(DG->tmp_fp, "Producer: Exit    Thread %ld ptr_in=%d ptr_out=%d ds_ptr=%d avg load time=%2.4f s. buffer_count=%d produced=%d consumed=%d\n", tid, DG->ptr_in, DG->ptr_out, DG->ds_ptr, 0.0, DG->buffer_count, DG->total_produced, DG->total_consumed);
                fflush(DG->tmp_fp);
        }
        //imprime_DG(__func__, DG);
    } 
 //   printf("%s exit\n", __func__);
   // fprintf(stdout, "6 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
    //fflush(stdout);
    pthread_exit(NULL);
    return NULL;
}

 // Create new buffer (dg_buffer==0) 
    // or enlarge it 
void create_dg_buffer(int size, const vector<int> shapeX,  const vector<int> shapeY) {
       
    printf("[DG] %s creating buffer: ", __func__); 
    for (int i = dg_buffer_size; i < size; i++) { 
        dg_bufferX[i] = new Tensor(shapeX); 
        dg_bufferY[i] = new Tensor(shapeY); 
        printf("%d ", i ); 
    } 
    if (size>dg_buffer_size)
        dg_buffer_size = size; 
    printf(". Buffer size: %d\n", dg_buffer_size ); 
    }
   
void* new_DataGen(DG_Data* Data, const char* filenameX, const char* filenameY, int dtype, int bs, bool distr_ds, int* dataset_size, int* nbpp,  int method, int num_threads, int buffer_size) {
    
    int check_ds_size = 0;
    size_t memsizeX = 0;
    size_t memsizeY = 0;
    char tmp_name[128];
    int id=get_id_distributed();
     int n_procs=get_n_procs_distributed();

     if (Data->created)
        msg("Error DG already created ", __func__);
    
    if (buffer_size > MAX_BUFFER)
        msg("Error buffer_size is too high ", __func__);
    Data->buffer_size=buffer_size;
    
    if (num_threads > MAX_DG_THREADS)
        msg("Error num threads is too high", __func__);
    Data->num_threads=num_threads;
    
    if ((total_dg>0) && (dg_batch_size!=bs))
         msg("Error All DG should have the same batch size", __func__);
    
    Data->dg_id = total_dg;
    total_dg++;
       
    Data->batch_size = bs;
    Data->distr_ds = distr_ds;
    
    if (Data->dg_id==0)
        dg_batch_size=Data->batch_size;
    
    Data->method=method;
    sprintf(Data->filenameX, "%s", filenameX);
    sprintf(Data->filenameY, "%s", filenameY);
    
    Data->datatype=dtype;
   

    printf("[DG] %s. datagen %d filenameX: %s filenameY: %s datatype= %d bs=%d distr_ds=%s method=%d num threads=%d buffer size=%d\n", __func__, Data->dg_id, Data->filenameX, Data->filenameY, Data->datatype, bs, (distr_ds==0?"no":"si"), Data->method, Data->num_threads, Data->buffer_size);

    if (is_mpi_distributed() == 0) {
        Data->fpX = fopen(Data->filenameX, "r");
        if (Data->fpX == NULL)
            msg("Error opening X file", __func__);
            fread(&Data->ndimX, sizeof (int), 1, Data->fpX);
    } else {
#ifdef cMPI
        MPI_File_open(MPI_COMM_WORLD, Data->filenameX, MPI_MODE_RDONLY, MPI_INFO_NULL, &Data->mfpX);
        if (Data->mfpX == NULL)
            msg("Error opening X file", __func__);
        MPI_File_read_all(Data->mfpX, &Data->ndimX, 1, MPI_INT, MPI_STATUS_IGNORE);
#endif
    }

    vector<int> r_shapeX(Data->ndimX);
    
    if (is_mpi_distributed() == 0) {
        fread(r_shapeX.data(), sizeof (int), Data->ndimX, Data->fpX);
    } else {
#ifdef cMPI
        MPI_File_read_all(Data->mfpX, r_shapeX.data(), Data->ndimX, MPI_INT, MPI_STATUS_IGNORE);
#endif 
    }
    Data->dataset_size = r_shapeX[0];
    *dataset_size =Data->dataset_size;

    Data->shape_sizeX = 1;
    for (int i = 1; i < Data->ndimX; i++) {
        Data->shape_sizeX *= r_shapeX[i];
    }
    // Force bs length in shape:
    r_shapeX[0] = Data->batch_size;

    Data->n_sizeX = Data->shape_sizeX*Data->batch_size;
 
    // Set nbpp
    Data->nbpp = set_NBPP_distributed(Data->dataset_size, Data->batch_size, Data->distr_ds);
    *nbpp = Data->nbpp;

    if (id == 0)
        printf("[DG] %s. filenameX: %s shape_sizeX=%d dataset_size=%d num_batches=%d\n", __func__, Data->filenameX, Data->shape_sizeX, Data->dataset_size, Data->nbpp);

    if (is_mpi_distributed() == 0) {
        Data->fpY = fopen(Data->filenameY, "r");
        if (Data->fpY == NULL)
            msg("Error opening Y file", __func__);
        fread(&Data->ndimY, sizeof (int), 1, Data->fpY);
    } else {
#ifdef cMPI
        MPI_File_open(MPI_COMM_WORLD, Data->filenameY, MPI_MODE_RDONLY, MPI_INFO_NULL, &Data->mfpY);
        if (Data->mfpY == NULL)
            msg("Error opening Y file", __func__);
        MPI_File_read_all(Data->mfpY, &Data->ndimY, 1, MPI_INT, MPI_STATUS_IGNORE);
#endif
    }
    
    vector<int> r_shapeY(Data->ndimY);
    if (is_mpi_distributed() == 0) {
       fread(r_shapeY.data(), sizeof (int), Data->ndimY, Data->fpY);
        
    } else {
#ifdef cMPI
        MPI_File_read_all(Data->mfpY, r_shapeY.data(), Data->ndimY, MPI_INT, MPI_STATUS_IGNORE);
#endif
    }
    check_ds_size = r_shapeY[0];
    Data->shape_sizeY = 1;
    for (int i = 1; i < Data->ndimY; i++) {
        Data->shape_sizeY *= r_shapeY[i];
    }
    // Force bs length in shape:
    r_shapeY[0] = Data->batch_size;
    Data->n_sizeY = Data->shape_sizeY*Data->batch_size;
  
    
    if (id == 0)
        printf("[DG] %s. filenameY: %s shape_sizeY=%d dataset_size=%d\n", __func__, Data->filenameY, Data->shape_sizeY, check_ds_size);
    
     if (Data->dataset_size != check_ds_size)
        msg("Error dataset sizes X and Y are different", __func__); // Exits


    //imprime_DG(__func__, Data);
    
  //  vector<int> aux_shapeX(Data->r_shapeX, Data->r_shapeX+Data->ndimX);
  //  vector<int> aux_shapeY(Data->r_shapeY, Data->r_shapeY+Data->ndimY);
    
    // Create buffer 
    create_dg_buffer(Data->buffer_size,r_shapeX, r_shapeY );
   
    /*
    if (total_dg==1) {
    for (int i = 0; i < Data->buffer_size; i++) {
        //Data->bufferX[i] = Tensor::empty(aux_shapeX, DEV_CPU);
        //Data->bufferY[i] = Tensor::empty(aux_shapeY, DEV_CPU);
        dg_bufferX[i] = new Tensor(aux_shapeX);
        dg_bufferY[i] = new Tensor(aux_shapeY);
    }
    }*/
   
    memsizeX = Data->buffer_size * Data->n_sizeX * sizeof (float);
    memsizeY = Data->buffer_size * Data->n_sizeY * sizeof (float);
    
     /*
      if (Data->perfect) {
        Data->list = new int[Data->nbpp * n_procs];
       //   dg_lista= new int[Data->nbpp * n_procs];
        if (Data->list == NULL)
            msg("Error in malloc (list)", __func__); // Exits
      //  printf("List address %0x\n", Data->list);
      //  for (int i=0; i< Data->nbpp * n_procs; i++)
      //      Data->list[i]=i;
    } 
    
    for (int i=0; i < 24; i++) {
        Data->list[i]=i;
	printf("%s %d\n", __func__, Data->list[i]);
    }
    */
    //Data->vlist.reserve(Data->nbpp * n_procs);
    
    if (id == 0)
        printf("[DG] %s. buffer requirements: bufferX= %.1fMB bufferY= %.1fMB \n", __func__, (float) memsizeX / (1024 * 1024), (float) memsizeY / (1024 * 1024));

    
    sprintf(tmp_name,"eddl_dg%d.log",Data->dg_id);
    //printf("%s\n",tmp_name);
    Data->tmp_fp = fopen(tmp_name, "w");
    if (Data->tmp_fp == NULL)
        msg("Error opening tmp file", __func__);
    
    Data->created=true;
    imprime_DG(__func__,Data);
    //imprime_buffer(Data);

    return NULL;
}

void* end_DataGen(DG_Data* DG) {
    int id=get_id_distributed();

    if (DG->created == false)
        msg("Error DG was not created ", __func__);
    if (DG->running)
        msg("Error DG is still running", __func__);

    if (is_mpi_distributed() == 0) {
        fclose(DG->fpX);
        fclose(DG->fpY);
    } else {
#ifdef cMPI
        MPICHECK(MPI_File_close(&DG->mfpX));
        MPICHECK(MPI_File_close(&DG->mfpY));
#endif
    }

    // Free buffer if it is the last one
    if (total_dg == 1) {
        for (int i = 0; i < dg_buffer_size; i++) {
            delete dg_bufferX[i];
            delete dg_bufferY[i];
        }
        if (id == 0)
            printf("[DG] %s datagen: %d. deleted %d entries\n", __func__, DG->dg_id, dg_buffer_size);
        dg_buffer_size = 0;
    }
    //  free(DG->list);
    fclose(DG->tmp_fp);
    if (id == 0)
        printf("[DG] %s datagen: %d \n", __func__, DG->dg_id);
    total_dg--;
    DG->created = false;
    return NULL;
}

void* start_DataGen(DG_Data* DG) {
    int err;
    int id=get_id_distributed();
     int n_procs=get_n_procs_distributed();
    
    if (nr_dg_running>0)
         msg("Error Only *one* DG can be started ", __func__);
    if (DG->created==false)        
        msg("Error DG was not created ", __func__);
    if (DG->running)
        msg("Error DG is already running", __func__);

    DG->buffer_count = 0;
    DG->total_produced =0;
    DG->total_consumed=0;
    DG->ptr_in = 0;
    DG->ptr_out = 0;
    DG->ds_ptr = 0;
    if (DG->method == DG_PERFECT) {     
        if (DG->distr_ds) {
            dg_lista = new int[DG->nbpp];
            gen_unique_random_list(dg_lista, DG->nbpp);
        } else {
            dg_lista = new int[DG->nbpp * n_procs];
            if (id == 0) {
                //gen_unique_random_vector(DG->vlist, DG->nbpp * n_procs);
                // gen_unique_random_vector(dg_vector, DG->nbpp * n_procs);
                //gen_unique_random_list(DG->list, DG->nbpp * n_procs);
                gen_unique_random_list(dg_lista, DG->nbpp * n_procs);
            }
#ifdef cMPI
            if (is_mpi_distributed())
                MPICHECK(MPI_Bcast(dg_lista, DG->nbpp * n_procs, MPI_INT, 0, MPI_COMM_WORLD));
#endif
        }
    }
    


    if (sem_init(&dmutex, 0, 1) != 0) exit(EXIT_FAILURE);
    if (sem_init(&llenar, 0, DG->buffer_size) != 0) exit(EXIT_FAILURE);
    if (sem_init(&vaciar, 0, 0) != 0) exit(EXIT_FAILURE);


    if (id == 0)
        printf("[DG] %s datagen: %d creating %d thread(s): ", __func__, DG->dg_id,DG->num_threads);
    for (int i = 0; i < DG->num_threads; i++) {
        if (id == 0)
            printf("%d ", i);

        err = pthread_create(&t[i], NULL, &producer_DataGen, DG);
        if (err) msg("Error creating thread", __func__); // Exits;
       
    }
    if (id == 0)
        printf("\n");
    DG->running=true;
    nr_dg_running++;
    return NULL;
}

void* stop_DataGen(DG_Data* DG) {
    int id=get_id_distributed();
    
    if (DG->created==false)
        msg("Error DG was not created ", __func__);
    if (DG->running==false)
        msg("Error DG is not running", __func__);
    for (int i = 0; i < DG->num_threads; i++)
        pthread_cancel(t[i]);
    for (int i = 0; i < DG->num_threads; i++)
        pthread_join(t[i], NULL);

    sem_destroy(&dmutex);
    sem_destroy(&llenar);
    sem_destroy(&vaciar);

     if (DG->method==DG_PERFECT) 
         delete [] dg_lista;


    

    if (id == 0)
        printf("[DG] %s datagen: %d count= %d\n", __func__, DG->dg_id,DG->buffer_count);   
    DG->running=false;
     nr_dg_running--;
     return NULL;
}


void* get_batch_DataGen(DG_Data* DG, Tensor* in, Tensor* out) {
    
    if (DG->created==false)
        msg("Error DG was not created ", __func__);
    if (DG->running==false)
        msg("Error DG is not running", __func__);
    
    sem_wait(&vaciar);

    //    fprintf(tmp_fp,"%s ptr_out %d count= %d\n", __func__, ptr_out, buffer_count);
    //    fflush(tmp_fp);  
    //   fprintf(stdout,"%s ptr_out %d count= %d\n", __func__, DG->ptr_out, DG->buffer_count);
    //   fflush(stdout);  


    #pragma omp parallel sections 
    {
        #pragma omp  section 
        //Tensor::copy(DG->bufferX[DG->ptr_out], in);
        Tensor::copy(dg_bufferX[DG->ptr_out], in);
        #pragma omp  section
        //Tensor::copy(DG->bufferY[DG->ptr_out], out);
        Tensor::copy(dg_bufferY[DG->ptr_out], out);
    }

    DG->ptr_out++;
    if (DG->ptr_out >= DG->buffer_size) DG->ptr_out = 0;

    sem_wait(&dmutex);
    DG->total_consumed++;
    DG->buffer_count--;
    sem_post(&dmutex);
    sem_post(&llenar);
    fprintf(DG->tmp_fp, "getBatch: ptr_in=%d ptr_out=%d ds_ptr=%d buffer_count=%d produced=%d consumed=%d\n", DG->ptr_in, DG->ptr_out, DG->ds_ptr, DG->buffer_count, DG->total_produced, DG->total_consumed);
    fflush(DG->tmp_fp);
    //imprime_DG(__func__,DG);
    return NULL;
}

void* imprime_DG(const char* titulo, DG_Data* DG) {
    printf("[DG] %s created %d running %d id %d files %s %s ndims %d %d shape %d %d nsize %d %d batch %d ds %d nbpp %d buffer %d threads %d count %d ptrin %d ptrout %d dsptr %d produced %d consumed %d\n", 
            titulo, DG->created, DG->running, DG->dg_id, DG->filenameX, DG->filenameY,
            DG->ndimX, DG->ndimY, DG->shape_sizeX, DG->shape_sizeY, DG->n_sizeX, DG->n_sizeY, DG->batch_size, DG->dataset_size, DG->nbpp, DG-> buffer_size, DG->num_threads, 
            DG->buffer_count, DG->ptr_in, DG->ptr_out, DG->ds_ptr, DG->total_produced, DG->total_consumed);  
     
     //gen_unique_random_list(DG->list, DG->nbpp * n_procs);
//for (int i=0; i < 24; i++) {
//	printf("%s %d\n", __func__, DG->list[i]);}
    return NULL;
}

void* imprime_buffer(DG_Data* DG) {
    for (int i=0; i<DG->buffer_size; i++) 
       // DG->bufferX[i]->info();
        dg_bufferX[i]->info();
    for (int i=0; i<DG->buffer_size; i++)
        //DG->bufferY[i]->info();
        dg_bufferY[i]->info();
    return NULL;
    
}

/// CLASS based DG
/*
void* producer_DG(void* arg) {
    bool run_producer = true;
    pid_t tid = pthread_self();
    double loadsecs=0;
    int curr_ptr;
    int curr_ds_ptr;
    DataGen* DG = (DataGen*) arg;
    int id=get_id_distributed();

    printf("%s tid=%d\n", __func__, tid);
    fflush(stdout);

    //   pthread_detach(tid);

    //__thread long record;
    while (run_producer) {
        // printf("==SIZES %d %d\n", DG->n_sizeX, DG->n_sizeY);
        sem_wait(&DG->llenar);
        sem_wait(&DG->dmutex);
        curr_ptr = DG->ptr_in;
        curr_ds_ptr = DG->ds_ptr;
        DG->ds_ptr++;
        if (DG->ds_ptr > DG->dg_nbpp)
            run_producer = false;
        //if (DEBUG)
        //  fprintf(stdout,"1 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
        //  fflush(stdout);
        //fprintf(DG->tmp_fp,"%s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
        //fflush(DG->tmp_fp);
        DG->ptr_in++;
        if (DG->ptr_in >= DG->dg_buffer_size) DG->ptr_in = 0;
        //     fprintf(stdout,"2 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
        //     fflush(stdout);
        sem_post(&DG->dmutex);
        if (run_producer) {
            TIME_POINT1(load);
            //       fprintf(stdout,"3 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
            //  fflush(stdout);
            if (is_mpi_distributed() == 0)
                DG->loadXY_perfect(curr_ptr, id * DG->dg_nbpp + curr_ds_ptr, DG->dg_perfect);
            else
                DG->loadXY_perfect_distr(curr_ptr, id * DG->dg_nbpp + curr_ds_ptr, DG->dg_perfect);
            TIME_POINT2(load, loadsecs);
            //        fprintf(stdout,"4 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
            //  fflush(stdout);
            //loadXY_Rand(curr_ptr);
            //record=rand() % dg_num_batches;
            //load(fpX, ndimX, shape_sizeX, record, bufferX, curr_ptr);
            //load(fpY, ndimY, shape_sizeY, record, bufferY, curr_ptr);
            sem_wait(&DG->dmutex);
            DG->buffer_count++;
            sem_post(&DG->dmutex);
            sem_post(&DG->vaciar);
        }
        //    fprintf(stdout,"5 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
        //   fflush(stdout);
        if ((DG->ds_ptr % 10) == 0) {
            sem_wait(&DG->imprimir);
            //    fprintf(tmp_fp, "Thread: %ld ptr_in=%d ptr_out=%d ds_ptr=%d avg load time=%2.4f s. buffer_count=%d\n", tid, DG->ptr_in, DG->ptr_out, DG->ds_ptr, (loadsecs*DG->dg_num_threads)/DG->ds_ptr, DG->buffer_count);
            //    fflush(DG->tmp_fp);
            sem_post(&DG->imprimir);
        }
    }
    printf("%s exit\n", __func__);
    fprintf(stdout, "6 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
    fflush(stdout);
    pthread_exit(NULL);
    return NULL;
}*/

/*
DataGen::DataGen(const string& filenameX, const string& filenameY, int bs, bool distr_ds, int* dataset_size, int* nbpp, bool perfect, int num_threads, int buffer_size) {
    int check_ds_size = 0;
    size_t memsizeX = 0;
    size_t memsizeY = 0;
    int id=get_id_distributed();
     int n_procs=get_n_procs_distributed();

    dg_id = total_dg;
    total_dg++;

    //if (dg_id>=DG_MAX)
    //     msg("Error number of data generators exceeded",__func__); 
    //printf("[DISTR] %s. filenameX: %s filenameY: %s bs=%d distr_ds=%s perfect=%d num threads=%d buffer size=%d\n",__func__, filenameX.c_str(), filenameY.c_str(), bs, (distr_ds==0?"no":"si"), perfect, num_threads, buffer_size);
    printf("[DISTR] %s. filenameX: %s filenameY: %s bs=%d distr_ds=%s perfect=%d num threads=%d buffer size=%d\n", __func__, filenameX.c_str(), filenameY.c_str(), bs, "no", perfect, num_threads, buffer_size);


    dg_perfect = perfect;
    dg_batch_size = bs;

    if (buffer_size > MAX_BUFFER)
        msg("Error buffer_size is too high ", __func__);
    dg_buffer_size = buffer_size;

    if (num_threads > MAX_DG_THREADS)
        msg("Error num threads is too high", __func__);
    dg_num_threads = num_threads;

    if (id == 0)
        printf("[DISTR] %s. perfect=%d buffer_size=%d. num_threads=%d. bs=%d. Using %s\n", __func__, dg_perfect, dg_buffer_size, dg_num_threads, dg_batch_size, "fopen()");
    //    printf("[DISTR] %s. perfect=%d buffer_size=%d. num_threads=%d. bs=%d. Using %s\n", __func__, dg_perfect, dg_buffer_size, dg_num_threads, dg_batch_size, (is_mpi_distributed()==0?"fopen()":"MPI_File_open()"));

    if (is_mpi_distributed() == 0) {
        fpX = fopen(filenameX.c_str(), "r");
        if (fpX == NULL)
            msg("Error opening X file", __func__);
        fread(&ndimX, sizeof (int), 1, fpX);
    } else {
#ifdef cMPI
        MPI_File_open(MPI_COMM_WORLD, filenameX.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &mfpX);
        if (mfpX == NULL)
            msg("Error opening X file", __func__);
        MPI_File_read_all(mfpX, &ndimX, 1, MPI_INT, MPI_STATUS_IGNORE);
#endif
    }

    vector<int> r_shapeX(ndimX);

    if (is_mpi_distributed() == 0)
        fread(r_shapeX.data(), sizeof (int), ndimX, fpX);
    else {
#ifdef cMPI
        MPI_File_read_all(mfpX, r_shapeX.data(), ndimX, MPI_INT, MPI_STATUS_IGNORE);
#endif
    }
    dg_dataset_size = r_shapeX[0];
    // *dataset_size = r_shapeX[0];
    //printf("%s %d\n", __func__, dataset_size);
    shape_sizeX = 1;
    for (int i = 1; i < ndimX; i++) {
        printf("shape %d = %d\n", i, r_shapeX[i]);
        shape_sizeX = shape_sizeX * r_shapeX[i];
    }
    // Force bs length in shape:
    r_shapeX[0] = dg_batch_size;

    printf("===ndimX %d shape_size %d batch %d\n", ndimX, shape_sizeX, dg_batch_size);

    int ojal = shape_sizeX*dg_batch_size;
    dg_n_sizeX = shape_sizeX*dg_batch_size;

    // Set nbpp
    dg_nbpp = set_NBPP_distributed(dg_dataset_size, bs, distr_ds);
    // *nbpp = dg_nbpp;

    if (id == 0)
        printf("[DISTR] %s. filenameX: %s shape_sizeX=%d dataset_size=%d num_batches=%d\n", __func__, filenameX.c_str(), shape_sizeX, dg_dataset_size, dg_nbpp);

    if (is_mpi_distributed() == 0) {
        fpY = fopen(filenameY.c_str(), "r");
        if (fpY == NULL)
            msg("Error opening Y file", __func__);
        fread(&ndimY, sizeof (int), 1, fpY);
    } else {
#ifdef cMPI
        MPI_File_open(MPI_COMM_WORLD, filenameY.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &mfpY);
        if (mfpY == NULL)
            msg("Error opening Y file", __func__);
        MPI_File_read_all(mfpY, &ndimY, 1, MPI_INT, MPI_STATUS_IGNORE);
#endif
    }
    vector<int> r_shapeY(ndimY);
    if (is_mpi_distributed() == 0)
        fread(r_shapeY.data(), sizeof (int), ndimY, fpY);
    else {
#ifdef cMPI
        MPI_File_read_all(mfpY, r_shapeY.data(), ndimY, MPI_INT, MPI_STATUS_IGNORE);
#endif
    }
    check_ds_size = r_shapeY[0];
    shape_sizeY = 1;
    for (int i = 1; i < ndimY; i++) {
        shape_sizeY *= r_shapeY[i];
    }
    // Force bs length in shape:
    r_shapeY[0] = dg_batch_size;
    dg_n_sizeY = shape_sizeY*dg_batch_size;

    if (id == 0)
        printf("[DISTR] %s. filenameY: %s shape_sizeY=%d dataset_size=%d\n", __func__, filenameY.c_str(), shape_sizeY, check_ds_size);

    for (int i = 0; i < dg_buffer_size; i++) {
        bufferX[i] = new Tensor(r_shapeX);
        bufferY[i] = new Tensor(r_shapeY);
    }
    memsizeX = dg_buffer_size * dg_n_sizeX * sizeof (float);
    memsizeY = dg_buffer_size * dg_n_sizeY * sizeof (float);

    if (dg_dataset_size != check_ds_size)
        msg("Error dataset sizes X and Y are different", __func__); // Exits

    if (dg_perfect) {
        this->list = (int*) malloc(dg_nbpp * n_procs * sizeof (int));
        if (this->list == NULL)
            msg("Error in malloc (list)", __func__); // Exits
    }

    if (id == 0)
        printf("[DISTR] %s. buffer requirements: bufferX= %.1fMB bufferY= %.1fMB list= %.1fMB \n", __func__, (float) memsizeX / (1024 * 1024), (float) memsizeY / (1024 * 1024), ((float) dg_nbpp * sizeof (int)) / (1024 * 1024));

#ifdef DEBUG_DONE
    
#endif
    srand(id * time(NULL));
    //printf("[DISTR] %s OK\n", __func__);
    
    //bytesX=(unsigned char*)malloc(shape_sizeX*dg_batch_size*sizeof(unsigned char));
    //if (bytesX==NULL)
    //    msg("Error in malloc (bytesX)", __func__); // Exits
    //bytesY=(unsigned char*)malloc(shape_sizeY*dg_batch_size*sizeof(unsigned char));
    // if (bytesY==NULL)
    //    msg("Error in malloc (bytesY)", __func__); // Exits
    
    //sprintf(tmp_name,"eddl_dg%d.log",dg_id);
    //printf("%s\n",tmp_name);
    if (dg_id == 0) {
        //tmp_fp= fopen(tmp_name, "w");
        //if (tmp_fp == NULL)
        //    msg("Error opening X file", __func__); 
    }

    printf("%s SIZES %d %d\n", __func__, n_sizeX, n_sizeY);
   
}

void* DataGen::end_data_generator() {
     if (dg_created == false)
        msg("Error DG was not created ", __func__);
    if (dg_running)
        msg("Error DG is still running", __func__);

    for (int i = 0; i < dg_buffer_size; i++) {
        delete this->bufferX[i];
        delete this->bufferY[i];
    }

    free(this->list);
    //   free(bytesX);
    //     free(bytesY);
    if (is_mpi_distributed() == 0) {
        fclose(fpX);
        fclose(fpY);
    } else {
#ifdef cMPI
        MPICHECK(MPI_File_close(&mfpX));
        MPICHECK(MPI_File_close(&mfpY));
#endif
    }
    //fclose(tmp_fp);  
}

void* DataGen::start_data_generator() {
    int tid;
    printf("%s datagen %d\n", __func__, dg_id);
    buffer_count = 0;
    ptr_in = 0;
    ptr_out = 0;
    ds_ptr = 0;
    int id=get_id_distributed();
     int n_procs=get_n_procs_distributed();
    
    if (dg_perfect) {
        if (id == 0)
            gen_unique_random_list(list, dg_nbpp * n_procs);
#ifdef cMPI
        if (is_mpi_distributed()) {
            MPICHECK(MPI_Bcast(list, dg_nbpp*n_procs, MPI_INT, 0, MPI_COMM_WORLD));
            printf("[DISTR] %s broadcast: ", __func__);
        }
#endif
    }
#ifdef DEBUG_DONE
    for (int i = 0; i < dg_nbpp; i++)
        done_batches[i] = 0;
    for (int i = 0; i < dg_dataset_size; i++)
        done_images[i] = 0;
#endif

    if (sem_init(&dmutex, 0, 1) != 0) exit(EXIT_FAILURE);
    if (sem_init(&llenar, 0, dg_buffer_size) != 0) exit(EXIT_FAILURE);
    if (sem_init(&vaciar, 0, 0) != 0) exit(EXIT_FAILURE);
    if (sem_init(&imprimir, 0, 1) != 0) exit(EXIT_FAILURE);



    if (id == 0)
        printf("[DISTR] %s creating %d thread(s): ", __func__, dg_num_threads);
    for (int i = 0; i < dg_num_threads; i++) {
        if (id == 0)
            printf("%d ", i);

        tid = pthread_create(&t[i], NULL, &producer_DG, this);
        if (tid) msg("Error creating thread", __func__); // Exits;       
        //pthread_detach(tid);
    }
    if (id == 0)
        printf("\n");
    printf("%s SIZES %d %d\n", __func__, dg_n_sizeX, dg_n_sizeY);
}


void* DataGen::stop_data_generator() {
    int id=get_id_distributed();
    
    for (int i = 0; i < dg_num_threads; i++)
        pthread_cancel(t[i]);
    //    for (int i = 0; i < dg_num_threads; i++)
    //        pthread_join(t[i], NULL);

    sem_destroy(&dmutex);
    sem_destroy(&llenar);
    sem_destroy(&vaciar);
    sem_destroy(&imprimir);



    if (id == 0)
        printf("[DISTR] %s count= %d\n", __func__, buffer_count);
}

void* DataGen::get_batch(Tensor* in, Tensor* out) {
    //printf("%s running\n", __func__);
    sem_wait(&vaciar);

    //    fprintf(tmp_fp,"%s ptr_out %d count= %d\n", __func__, ptr_out, buffer_count);
    //    fflush(tmp_fp);  
    //   fprintf(stdout,"%s ptr_out %d count= %d\n", __func__, ptr_out, buffer_count);
    //   fflush(stdout);  


    //#pragma omp parallel sections 
    {
        //#pragma omp  section 
        Tensor::copy(this->bufferX[ptr_out], in);
        //#pragma omp  section
        Tensor::copy(this->bufferY[ptr_out], out);
    }

    ptr_out++;
    if (ptr_out >= dg_buffer_size) ptr_out = 0;

    sem_wait(&dmutex);
    buffer_count--;
    sem_post(&dmutex);
    sem_post(&llenar);
}

int DataGen::get_buffer_count() {
    return buffer_count;
}

int DataGen::get_nbpp() {
    return dg_nbpp;
}

int DataGen::get_dataset_size() {
    return dg_dataset_size;
}

void DataGen::loadXY_perfect(int buffer_index, int ds_ptr, bool perfect) {
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

    //  printf("SIZES %d %d\n", n_sizeX, this->n_sizeY);
    // Random batches of sequential items
    bytesX = (unsigned char*) malloc(this->n_sizeX);
    if (bytesX == NULL)
        msg("Error bytesX memory allocation", __func__);
    bytesY = (unsigned char*) malloc(this->n_sizeY);
    if (bytesY == NULL)
        msg("Error bytesY memory allocation", __func__);


    //  printf("1 %s ds_ptr=%ld\n", __func__, ds_ptr);
    if (perfect)
        pos = list[ds_ptr];

    else {
        pos = rand() % dg_nbpp;
    }

    // fprintf(tmp_fp,"%s sizeof(size_t)=%d perfect=%d buffer_index=%d num_batches=%d sizes= %ld %ld pos=%d\n", __func__, sizeof(size_t), perfect, buffer_index, dg_num_batches, n_sizeX, n_sizeY,pos);
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
        printf("%s n_read %d n_size %d\n", __func__, n_read, n_sizeX);
        msg("Error freadX ", __func__);
    }
    // printf("2 %s ds_ptr=%ld\n", __func__, ds_ptr);


    //printf("%s count=%d buffer_index=%d ptr_out=%d pos=%ld\n",__func__,buffer_count,buffer_index,ptr_out,pos);
    if (DEBUG)
        printf("LOAD:");
    //#pragma omp parallel for private(j,index)  
    for (i = 0; i < dg_batch_size; i++) {
        for (j = 0; j < shape_sizeX; j++) {
            index = i * shape_sizeX + j;
            bufferX[buffer_index]->ptr[index] = (float) bytesX[index];
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
        printf("%s n_read %d n_size %d\n", __func__, n_read, n_sizeY);
        msg("Error freadY ", __func__);
    }
    //    printf("3 %s ds_ptr=%ld\n", __func__, ds_ptr);

    if (DEBUG) printf("LOAD:");
    // #pragma omp parallel for private(j,index) 
    for (i = 0; i < dg_batch_size; i++) {
        for (j = 0; j < shape_sizeY; j++) {
            index = i * shape_sizeY + j;
            bufferY[buffer_index]->ptr[index] = (float) bytesY[index];
            //            bufferY[index] = (float) 0;
            //if (DEBUG) 
            //  printf("[%d %3.1f ] ", index, bufferY[index]);
        }
    }
    //      printf("4 %s ds_ptr=%ld\n", __func__, ds_ptr);
    if (DEBUG) printf("\n");
    free(bytesX);
    free(bytesY);
    //    printf("5 %s ds_ptr=%ld\n", __func__, ds_ptr);
}

void DataGen::loadXY_perfect_distr(int buffer_index, int ds_ptr, bool perfect) {
#ifdef cMPI
    //    unsigned char bytesX[n_sizeX];
    //    unsigned char bytesY[n_sizeY];
    unsigned char* bytesX;
    unsigned char* bytesY;
    long i, j;
    int index;
    int err;
    //long int pos;
    MPI_Offset pos;
    MPI_Status status;
    off_t posX, posY;
    int n_read;


    // Random batches of sequential items
    bytesX = (unsigned char*) malloc(n_sizeX);
    if (bytesX == NULL)
        msg("Error bytesX memory allocation", __func__);
    bytesY = (unsigned char*) malloc(n_sizeY);
    if (bytesY == NULL)
        msg("Error bytesY memory allocation", __func__);


    //printf("%s ds_ptr=%ld", __func__, ds_ptr);
    if (perfect)
        pos = list[ds_ptr];
    else {
        pos = rand() % dg_nbpp;
    }
    //fprintf(tmp_fp,"%s sizeof(size_t)=%d perfect=%d buffer_index=%d num_batches=%d sizes= %ld %ld pos=%d\n", __func__, sizeof(size_t), perfect, buffer_index, dg_nbpp, n_sizeX, n_sizeY,pos);
    //fflush(tmp_fp);
    //pos=0;

#ifdef DEBUG_DONE
    done_batches[pos] += 1;
#endif 
    posX = (off_t) (ndimX + 1) * sizeof (int)+(off_t) pos * n_sizeX * sizeof (unsigned char);
    //    fprintf(tmp_fp,"%s ds_ptr=%d pos=%ld  \n", __func__, ds_ptr, posX);
    //    fflush(tmp_fp);

    MPICHECK(MPI_File_seek(mfpX, posX, MPI_SEEK_SET));
    MPICHECK(MPI_File_read(mfpX, bytesX, n_sizeX, MPI_BYTE, &status));
    MPICHECK(MPI_Get_count(&status, MPI_BYTE, &n_read));

    if (n_read != n_sizeX) {
        printf("%s n_read %d n_size %d\n", __func__, n_read, n_sizeX);
        msg("Error freadX ", __func__);
    }

    //printf("%s count=%d buffer_index=%d ptr_out=%d pos=%ld\n",__func__,buffer_count,buffer_index,ptr_out,pos);
    if (DEBUG)
        printf("LOAD:");
    //#pragma omp parallel for private(j,index)  
    for (i = 0; i < dg_batch_size; i++) {
        for (j = 0; j < shape_sizeX; j++) {
            index = i * shape_sizeX + j;
            //printf("%s buffer_index=%d index=%d\n",__func__,buffer_index,index);
            bufferX[buffer_index]->ptr[index] = (float) bytesX[index];
            //if (DEBUG)
            // printf("[%d %3.1f ] ", index, bufferX[index]);
        }
    }
    if (DEBUG) printf("\n");
    posY = (off_t) (ndimY + 1) * sizeof (int)+(off_t) pos * n_sizeY * sizeof (unsigned char);

    //MPI_File_read_at(mfpY, posY, bytesY, n_sizeY, MPI_BYTE, MPI_STATUS_IGNORE);
    MPICHECK(MPI_File_seek(mfpY, posY, MPI_SEEK_SET));
    MPICHECK(MPI_File_read(mfpY, bytesY, n_sizeY, MPI_BYTE, &status));
    MPICHECK(MPI_Get_count(&status, MPI_BYTE, &n_read));

    if (n_read != n_sizeY) {
        printf("%s n_read %d n_size %d\n", __func__, n_read, n_sizeY);
        msg("Error freadY ", __func__);
    }
    //fprintf(tmp_fp,"process Y %d read %d bytes\n", id, n_sizeY, count);


    if (DEBUG) printf("LOAD:");
    //#pragma omp parallel for private(j,index) 
    for (i = 0; i < dg_batch_size; i++) {
        for (j = 0; j < shape_sizeY; j++) {
            index = i * shape_sizeY + j;
            bufferY[buffer_index]->ptr[index] = (float) bytesY[index];
            //            bufferY[index] = (float) 0;
            //if (DEBUG) 
            //  printf("[%d %3.1f ] ", index, bufferY[index]);
        }
    }
    if (DEBUG) printf("\n");
    free(bytesX);
    free(bytesY);
#endif
}

DataGen2::DataGen2(const string& filenameX, const string& filenameY, int bs, bool distr_ds, int* dataset_size, int* nbpp, bool perfect, int num_threads, int buffer_size) {

    int check_ds_size = 0;
    size_t memsizeX = 0;
    size_t memsizeY = 0;
    size_t oj;
    
    int id=get_id_distributed();
     int n_procs=get_n_procs_distributed();

    dg_id = total_dg;
    total_dg++;


    //if (dg_id>=DG_MAX)
    //     msg("Error number of data generators exceeded",__func__); 
    printf("[DISTR] %s. datagen %d filenameX: %s filenameY: %s bs=%d distr_ds=%s perfect=%d num threads=%d buffer size=%d\n", __func__, dg_id, filenameX.c_str(), filenameY.c_str(), bs, "no", perfect, num_threads, buffer_size);



    dg_perfect = perfect;
    dg_batch_size = bs;

    if (buffer_size > MAX_BUFFER)
        msg("Error buffer_size is too high ", __func__);
    dg_buffer_size = buffer_size;

    if (num_threads > MAX_DG_THREADS)
        msg("Error num threads is too high", __func__);
    dg_num_threads = num_threads;

    if (id == 0)
        printf("[DISTR] %s. perfect=%d buffer_size=%d. num_threads=%d. bs=%d. Using %s\n", __func__, dg_perfect, dg_buffer_size, dg_num_threads, dg_batch_size, "no");
  
    if (is_mpi_distributed() == 0) {
        fpX = fopen(filenameX.c_str(), "r");
        if (fpX == NULL)
            msg("Error opening X file", __func__);
        oj = fread(&ndimX, sizeof (int), 1, fpX);
        printf("READX %d \n", oj);
    } else {
#ifdef cMPI
        MPI_File_open(MPI_COMM_WORLD, filenameX.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &mfpX);
        if (mfpX == NULL)
            msg("Error opening X file", __func__);
        MPI_File_read_all(mfpX, &ndimX, 1, MPI_INT, MPI_STATUS_IGNORE);
#endif
    }

    vector<int> r_shapeX(ndimX);

    if (is_mpi_distributed() == 0) {
        oj = fread(r_shapeX.data(), sizeof (int), ndimX, fpX);
        printf("READ shaoe X %d \n", oj);
    } else {
#ifdef cMPI
        MPI_File_read_all(mfpX, r_shapeX.data(), ndimX, MPI_INT, MPI_STATUS_IGNORE);
#endif 
    }
    m_dataset_size = r_shapeX[0];
    *dataset_size = m_dataset_size;

    printf("=== despues %s %d %d \n", __func__, r_shapeX[0], m_dataset_size);
    shape_sizeX = 1;
    for (int i = 1; i < ndimX; i++) {
        printf("shape %d = %d\n", i, r_shapeX[i]);
        shape_sizeX = shape_sizeX * r_shapeX[i];
    }
    // Force bs length in shape:
    r_shapeX[0] = dg_batch_size;

    printf("===ndimX %d shape_size %d batch %d\n", ndimX, shape_sizeX, dg_batch_size);

    dg_n_sizeX = shape_sizeX*dg_batch_size;

    // Set nbpp
    dg_nbpp = set_NBPP_distributed(m_dataset_size, bs, distr_ds);
    *nbpp = dg_nbpp;

    if (id == 0)
        printf("[DISTR] %s. filenameX: %s shape_sizeX=%d dataset_size=%d num_batches=%d\n", __func__, filenameX.c_str(), shape_sizeX, m_dataset_size, dg_nbpp);

    if (is_mpi_distributed() == 0) {
        fpY = fopen(filenameY.c_str(), "r");
        if (fpY == NULL)
            msg("Error opening Y file", __func__);
        fread(&ndimY, sizeof (int), 1, fpY);
    } else {
#ifdef cMPI
        MPI_File_open(MPI_COMM_WORLD, filenameY.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &mfpY);
        if (mfpY == NULL)
            msg("Error opening Y file", __func__);
        MPI_File_read_all(mfpY, &ndimY, 1, MPI_INT, MPI_STATUS_IGNORE);
#endif
    }
    vector<int> r_shapeY(ndimY);
    if (is_mpi_distributed() == 0) {
        oj = fread(r_shapeY.data(), sizeof (int), ndimY, fpY);
        printf("READY %d \n", oj);
    } else {
#ifdef cMPI
        MPI_File_read_all(mfpY, r_shapeY.data(), ndimY, MPI_INT, MPI_STATUS_IGNORE);
#endif
    }
    check_ds_size = r_shapeY[0];
    shape_sizeY = 1;
    for (int i = 1; i < ndimY; i++) {
        shape_sizeY *= r_shapeY[i];
    }
    // Force bs length in shape:
    r_shapeY[0] = dg_batch_size;
    dg_n_sizeY = shape_sizeY*dg_batch_size;

    if (id == 0)
        printf("[DISTR] %s. filenameY: %s shape_sizeY=%d dataset_size=%d\n", __func__, filenameY.c_str(), shape_sizeY, check_ds_size);

    for (int i = 0; i < dg_buffer_size; i++) {
        bufferX[i] = new Tensor(r_shapeX);
        bufferY[i] = new Tensor(r_shapeY);
    }
    memsizeX = dg_buffer_size * dg_n_sizeX * sizeof (float);
    memsizeY = dg_buffer_size * dg_n_sizeY * sizeof (float);

    if (m_dataset_size != check_ds_size)
        msg("Error dataset sizes X and Y are different", __func__); // Exits

    if (dg_perfect) {
        this->list = (int*) malloc(dg_nbpp * n_procs * sizeof (int));
        if (this->list == NULL)
            msg("Error in malloc (list)", __func__); // Exits
    }

    if (id == 0)
        printf("[DISTR] %s. buffer requirements: bufferX= %.1fMB bufferY= %.1fMB list= %.1fMB \n", __func__, (float) memsizeX / (1024 * 1024), (float) memsizeY / (1024 * 1024), ((float) dg_nbpp * sizeof (int)) / (1024 * 1024));
}

void* producer_F(void* arg) {
    bool run_producer = true;
    pid_t tid = pthread_self();
    double loadsecs=0;
    int curr_ptr;
    int curr_ds_ptr;
    int id=get_id_distributed();
    DataGen2* DG = (DataGen2*) arg;


    printf("%s tid=%ld\n", __func__, tid);
    fflush(stdout);

    //   pthread_detach(tid);

    //__thread long record;
    while (run_producer) {
        // printf("==SIZES %d %d\n", DG->n_sizeX, DG->n_sizeY);
        sem_wait(&DG->llenar);
        sem_wait(&DG->dmutex);
        curr_ptr = DG->ptr_in;
        curr_ds_ptr = DG->ds_ptr;
        DG->ds_ptr++;
        if (DG->ds_ptr > DG->dg_nbpp)
            run_producer = false;
        //if (DEBUG)
        //  fprintf(stdout,"1 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
        //  fflush(stdout);
        //fprintf(DG->tmp_fp,"%s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
        //fflush(DG->tmp_fp);
        DG->ptr_in++;
        if (DG->ptr_in >= DG->dg_buffer_size) DG->ptr_in = 0;
        //     fprintf(stdout,"2 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
        //     fflush(stdout);
        sem_post(&DG->dmutex);
        if (run_producer) {
            TIME_POINT1(load);
            //       fprintf(stdout,"3 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
            //  fflush(stdout);
            if (is_mpi_distributed() == 0)
                DG->loadXY_perfect(curr_ptr, id * DG->dg_nbpp + curr_ds_ptr, DG->dg_perfect);
            else
                DG->loadXY_perfect_distr(curr_ptr, id * DG->dg_nbpp + curr_ds_ptr, DG->dg_perfect);
            TIME_POINT2(load, loadsecs);
            //        fprintf(stdout,"4 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
            //  fflush(stdout);
            //loadXY_Rand(curr_ptr);
            //record=rand() % dg_num_batches;
            //load(fpX, ndimX, shape_sizeX, record, bufferX, curr_ptr);
            //load(fpY, ndimY, shape_sizeY, record, bufferY, curr_ptr);
            sem_wait(&DG->dmutex);
            DG->buffer_count++;
            sem_post(&DG->dmutex);
            sem_post(&DG->vaciar);
        }
        //    fprintf(stdout,"5 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
        //   fflush(stdout);
        if ((DG->ds_ptr % 10) == 0) {
            sem_wait(&DG->imprimir);
            //    fprintf(tmp_fp, "Thread: %ld ptr_in=%d ptr_out=%d ds_ptr=%d avg load time=%2.4f s. buffer_count=%d\n", tid, DG->ptr_in, DG->ptr_out, DG->ds_ptr, (loadsecs*DG->dg_num_threads)/DG->ds_ptr, DG->buffer_count);
            //    fflush(DG->tmp_fp);
            sem_post(&DG->imprimir);
        }
    }
    printf("%s exit\n", __func__);
    fprintf(stdout, "6 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
    fflush(stdout);
    pthread_exit(NULL);
}

void DataGen2::start_data_generator() {
    int tid;
    printf("%s datagen %d\n", __func__, dg_id);
    buffer_count = 0;
    ptr_in = 0;
    ptr_out = 0;
    ds_ptr = 0;
    int id=get_id_distributed();
     int n_procs=get_n_procs_distributed();
    
    if (dg_perfect) {
        if (id == 0)
            gen_unique_random_list(list, dg_nbpp * n_procs);
#ifdef cMPI
        if (is_mpi_distributed()) {
            MPICHECK(MPI_Bcast(list, dg_nbpp*n_procs, MPI_INT, 0, MPI_COMM_WORLD));
            printf("[DISTR] %s broadcast: ", __func__);
        }
#endif
    }
#ifdef DEBUG_DONE
    for (int i = 0; i < dg_nbpp; i++)
        done_batches[i] = 0;
    for (int i = 0; i < dg_dataset_size; i++)
        done_images[i] = 0;
#endif

    if (sem_init(&dmutex, 0, 1) != 0) exit(EXIT_FAILURE);
    if (sem_init(&llenar, 0, dg_buffer_size) != 0) exit(EXIT_FAILURE);
    if (sem_init(&vaciar, 0, 0) != 0) exit(EXIT_FAILURE);
    if (sem_init(&imprimir, 0, 1) != 0) exit(EXIT_FAILURE);



    if (id == 0)
        printf("[DISTR] %s creating %d thread(s): ", __func__, dg_num_threads);
    for (int i = 0; i < dg_num_threads; i++) {
        if (id == 0)
            printf("%d ", i);

        //tid= pthread_create(&t[i], NULL, InternalThreadEntryFunc, this);
        tid = pthread_create(&t[i], NULL, &producer_F, this);
        if (tid) msg("Error creating thread", __func__); // Exits;       
        //pthread_detach(tid);
    }
    if (id == 0)
        printf("\n");
    printf("%s SIZES %d %d\n", __func__, n_sizeX, n_sizeY);
}

void DataGen2::stop_data_generator() {
    int id=get_id_distributed();
    
    for (int i = 0; i < dg_num_threads; i++)
        pthread_cancel(t[i]);
    //    for (int i = 0; i < dg_num_threads; i++)
    //        pthread_join(t[i], NULL);

    sem_destroy(&dmutex);
    sem_destroy(&llenar);
    sem_destroy(&vaciar);
    sem_destroy(&imprimir);



    if (id == 0)
        printf("[DISTR] %s count= %d\n", __func__, buffer_count);
}

void DataGen2::get_batch(Tensor* in, Tensor* out) {
    //printf("%s running\n", __func__);
    sem_wait(&vaciar);

    //    fprintf(tmp_fp,"%s ptr_out %d count= %d\n", __func__, ptr_out, buffer_count);
    //    fflush(tmp_fp);  
    //   fprintf(stdout,"%s ptr_out %d count= %d\n", __func__, ptr_out, buffer_count);
    //   fflush(stdout);  


    //#pragma omp parallel sections 
    {
        //#pragma omp  section 
        Tensor::copy(this->bufferX[ptr_out], in);
        //#pragma omp  section
        Tensor::copy(this->bufferY[ptr_out], out);
    }

    ptr_out++;
    if (ptr_out >= dg_buffer_size) ptr_out = 0;

    sem_wait(&dmutex);
    buffer_count--;
    sem_post(&dmutex);
    sem_post(&llenar);
}

void DataGen2::loadXY_perfect(int buffer_index, int ds_ptr, bool perfect) {
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

    //  printf("SIZES %d %d\n", n_sizeX, this->n_sizeY);
    // Random batches of sequential items
    bytesX = (unsigned char*) malloc(this->n_sizeX);
    if (bytesX == NULL)
        msg("Error bytesX memory allocation", __func__);
    bytesY = (unsigned char*) malloc(this->n_sizeY);
    if (bytesY == NULL)
        msg("Error bytesY memory allocation", __func__);


    //  printf("1 %s ds_ptr=%ld\n", __func__, ds_ptr);
    if (perfect)
        pos = list[ds_ptr];

    else {
        pos = rand() % dg_nbpp;
    }

    // fprintf(tmp_fp,"%s sizeof(size_t)=%d perfect=%d buffer_index=%d num_batches=%d sizes= %ld %ld pos=%d\n", __func__, sizeof(size_t), perfect, buffer_index, dg_num_batches, n_sizeX, n_sizeY,pos);
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
        printf("%s n_read %d n_size %d\n", __func__, n_read, n_sizeX);
        msg("Error freadX ", __func__);
    }
    // printf("2 %s ds_ptr=%ld\n", __func__, ds_ptr);


    //printf("%s count=%d buffer_index=%d ptr_out=%d pos=%ld\n",__func__,buffer_count,buffer_index,ptr_out,pos);
    if (DEBUG)
        printf("LOAD:");
    //#pragma omp parallel for private(j,index)  
    for (i = 0; i < dg_batch_size; i++) {
        for (j = 0; j < shape_sizeX; j++) {
            index = i * shape_sizeX + j;
            bufferX[buffer_index]->ptr[index] = (float) bytesX[index];
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
        printf("%s n_read %ld n_size %d\n", __func__, n_read, n_sizeY);
        msg("Error freadY ", __func__);
   }
    //    printf("3 %s ds_ptr=%ld\n", __func__, ds_ptr);

    if (DEBUG) printf("LOAD:");
    // #pragma omp parallel for private(j,index) 
    for (i = 0; i < dg_batch_size; i++) {
        for (j = 0; j < shape_sizeY; j++) {
            index = i * shape_sizeY + j;
            bufferY[buffer_index]->ptr[index] = (float) bytesY[index];
            //            bufferY[index] = (float) 0;
            //if (DEBUG) 
            //  printf("[%d %3.1f ] ", index, bufferY[index]);
        }
    }
    //      printf("4 %s ds_ptr=%ld\n", __func__, ds_ptr);
    if (DEBUG) printf("\n");
    free(bytesX);
    free(bytesY);
    //    printf("5 %s ds_ptr=%ld\n", __func__, ds_ptr);
}

void DataGen2::loadXY_perfect_distr(int buffer_index, int ds_ptr, bool perfect) {
#ifdef cMPI
    //    unsigned char bytesX[n_sizeX];
    //    unsigned char bytesY[n_sizeY];
    unsigned char* bytesX;
    unsigned char* bytesY;
    long i, j;
    int index;
    int err;
    //long int pos;
    MPI_Offset pos;
    MPI_Status status;
    off_t posX, posY;
    int n_read;


    // Random batches of sequential items
    bytesX = (unsigned char*) malloc(n_sizeX);
    if (bytesX == NULL)
        msg("Error bytesX memory allocation", __func__);
    bytesY = (unsigned char*) malloc(n_sizeY);
    if (bytesY == NULL)
        msg("Error bytesY memory allocation", __func__);


    //printf("%s ds_ptr=%ld", __func__, ds_ptr);
    if (perfect)
        pos = list[ds_ptr];
    else {
        pos = rand() % dg_nbpp;
    }
    //fprintf(tmp_fp,"%s sizeof(size_t)=%d perfect=%d buffer_index=%d num_batches=%d sizes= %ld %ld pos=%d\n", __func__, sizeof(size_t), perfect, buffer_index, dg_nbpp, n_sizeX, n_sizeY,pos);
    //fflush(tmp_fp);
    //pos=0;

#ifdef DEBUG_DONE
    done_batches[pos] += 1;
    done_images[pos] += 1;
#endif 
    posX = (off_t) (ndimX + 1) * sizeof (int)+(off_t) pos * n_sizeX * sizeof (unsigned char);
    //    fprintf(tmp_fp,"%s ds_ptr=%d pos=%ld  \n", __func__, ds_ptr, posX);
    //    fflush(tmp_fp);

    MPICHECK(MPI_File_seek(mfpX, posX, MPI_SEEK_SET));
    MPICHECK(MPI_File_read(mfpX, bytesX, n_sizeX, MPI_BYTE, &status));
    MPICHECK(MPI_Get_count(&status, MPI_BYTE, &n_read));

    if (n_read != n_sizeX) {
        printf("%s n_read %d n_size %d\n", __func__, n_read, n_sizeX);
        msg("Error freadX ", __func__);
    }

    //printf("%s count=%d buffer_index=%d ptr_out=%d pos=%ld\n",__func__,buffer_count,buffer_index,ptr_out,pos);
    if (DEBUG)
        printf("LOAD:");
    //#pragma omp parallel for private(j,index)  
    for (i = 0; i < dg_batch_size; i++) {
        for (j = 0; j < shape_sizeX; j++) {
            index = i * shape_sizeX + j;
            //printf("%s buffer_index=%d index=%d\n",__func__,buffer_index,index);
            bufferX[buffer_index]->ptr[index] = (float) bytesX[index];
            //if (DEBUG)
            // printf("[%d %3.1f ] ", index, bufferX[index]);
        }
    }
    if (DEBUG) printf("\n");
    posY = (off_t) (ndimY + 1) * sizeof (int)+(off_t) pos * n_sizeY * sizeof (unsigned char);

    //MPI_File_read_at(mfpY, posY, bytesY, n_sizeY, MPI_BYTE, MPI_STATUS_IGNORE);
    MPICHECK(MPI_File_seek(mfpY, posY, MPI_SEEK_SET));
    MPICHECK(MPI_File_read(mfpY, bytesY, n_sizeY, MPI_BYTE, &status));
    MPICHECK(MPI_Get_count(&status, MPI_BYTE, &n_read));

    if (n_read != n_sizeY) {
        printf("%s n_read %d n_size %d\n", __func__, n_read, n_sizeY);
        msg("Error freadY ", __func__);
    }
    //fprintf(tmp_fp,"process Y %d read %d bytes\n", id, n_sizeY, count);


    if (DEBUG) printf("LOAD:");
    //#pragma omp parallel for private(j,index) 
    for (i = 0; i < dg_batch_size; i++) {
        for (j = 0; j < shape_sizeY; j++) {
            index = i * shape_sizeY + j;
            bufferY[buffer_index]->ptr[index] = (float) bytesY[index];
            //            bufferY[index] = (float) 0;
            //if (DEBUG) 
            //  printf("[%d %3.1f ] ", index, bufferY[index]);
        }
    }
    if (DEBUG) printf("\n");
    free(bytesX);
    free(bytesY);
#endif
}
/*
void DataGen2::InternalThreadEntry() 
 {
    bool run_producer = true;
    pid_t tid = pthread_self();
    double loadsecs;
    int curr_ptr;
    int curr_ds_ptr;

printf("%s tid=%ld\n", __func__, tid);
fflush(stdout);

 //   pthread_detach(tid);
    
    //__thread long record;
    while (run_producer) {
        // printf("==SIZES %d %d\n", DG->n_sizeX, DG->n_sizeY);
        sem_wait(&llenar);
        sem_wait(&dmutex);
        curr_ptr = ptr_in;
        curr_ds_ptr = ds_ptr;
        ds_ptr++;
        if (ds_ptr > dg_nbpp)
            run_producer = false;
        //if (DEBUG)
      //  fprintf(stdout,"1 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
      //  fflush(stdout);
        //fprintf(DG->tmp_fp,"%s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
        //fflush(DG->tmp_fp);
        ptr_in++;
        if (ptr_in >= dg_buffer_size) ptr_in = 0;
    //     fprintf(stdout,"2 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
   //     fflush(stdout);
        sem_post(&dmutex);
        if (run_producer) {
            TIME_POINT1(load);
      //       fprintf(stdout,"3 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
      //  fflush(stdout);
            if (is_mpi_distributed()==0)
                loadXY_perfect(curr_ptr, id*dg_nbpp+curr_ds_ptr, dg_perfect);
            else
                loadXY_perfect_distr(curr_ptr, id*dg_nbpp+curr_ds_ptr, dg_perfect);
            TIME_POINT2(load, loadsecs);
     //        fprintf(stdout,"4 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
      //  fflush(stdout);
            //loadXY_Rand(curr_ptr);
            //record=rand() % dg_num_batches;
            //load(fpX, ndimX, shape_sizeX, record, bufferX, curr_ptr);
            //load(fpY, ndimY, shape_sizeY, record, bufferY, curr_ptr);
            sem_wait(&dmutex);
            buffer_count++;
            sem_post(&dmutex);
            sem_post(&vaciar);
        }
    //    fprintf(stdout,"5 %s ptr_in %d count= %d\n", __func__, DG->ptr_in, DG->buffer_count);
     //   fflush(stdout);
        if ((ds_ptr % 10) == 0) {
            sem_wait(&imprimir);
        //    fprintf(tmp_fp, "Thread: %ld ptr_in=%d ptr_out=%d ds_ptr=%d avg load time=%2.4f s. buffer_count=%d\n", tid, DG->ptr_in, DG->ptr_out, DG->ds_ptr, (loadsecs*DG->dg_num_threads)/DG->ds_ptr, DG->buffer_count);
        //    fflush(DG->tmp_fp);
            sem_post(&imprimir);
        }
    }
    printf("%s exit\n",__func__);
    fprintf(stdout,"6 %s ptr_in %d count= %d\n", __func__, ptr_in, buffer_count);
        fflush(stdout);
    pthread_exit(NULL);
}
 */
		
