/*
* MPI support for EDDL Library - European Distributed Deep Learning Library.
* Version: 
* copyright (c) 2021, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: July 2021
* Author: GAP Research Group (UPV), contact: plopez@disca.upv.es
* All rights reserved
*/



#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H



#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include "eddl/apis/eddl.h"
#include "omp.h"
#include <pthread.h>
#include <semaphore.h>



#ifdef cMPI
#include <mpi.h>

#endif



#define DG_MAX 4
#define DG_TRAIN 0
#define DG_VAL 1
#define DG_TEST 2
#define DG_USER 3

#define MAX_BUFFER 16
#define MAX_DG_THREADS 8

#define DG_RANDOM 0
#define DG_PERFECT 1
#define DG_LIN 2




/**
 *  @brief Initializes data generator
 *
 *  @param [in] dg_id           Id of data generator. Only *one* data generator is allowed. 
 *  @param [in] filenameX       Name of dataset file with samples
 *  @param [in] filenameY       Name of dataset file with labels
 *  @param [in] ds              (global) Batch size
 *  @param [in] distr_ds        DISTR_DS or NO_DISTR_DS
 *  @param [out] dataset_size   Dataset size
 *  @param [out] nbpp           Nr of batches per proc
 *  @param [in] method          DG_LIN, DG_RANDOM or DG_PERFECT
 *  @param [in] num_threads     Nr of threads to load samples
 *  @param [in] buffer_size     Size of buffer in (local) batches
 */
void* prepare_data_generator(int dg_id, const string &filenameX, const string &filenameY, int bs, bool distr_ds, int* dataset_size, int* nbpp,  int method, int num_threads, int buffer_size);

/**
 *  @brief Starts data generator
 */
void* start_data_generator(); 

/**
 *  @brief Stops data generator
 */
void* stop_data_generator();

/**
 *  @brief Frees data structures of data generator 
 */
void* end_data_generator();

/**
 *  @brief Copy next batch from buffer
 * 
 *  @param [out] in     Tensor of batch samples
 *  @param [out] out    Tensor of batch labels
 */
void* get_batch(Tensor* in, Tensor* out);

int get_buffer_count();



struct DG_Data {
    int dg_id;
   
    bool created=false;
    bool running=false;
   
    int ndimX=0;
    int ndimY=0;
    int shape_sizeX=0;
    int shape_sizeY=0;
    int n_sizeX=0;
    int n_sizeY=0;
   
    int batch_size=0;
    int dataset_size=0;
    int nbpp=0;
    int method=0;
    bool distr_ds = false;
    int buffer_size=0;
    int num_threads=0;
    
    int buffer_count = 0;
    int ptr_in = 0;
    int ptr_out = 0;
    int ds_ptr = 0;
    
    int total_produced=0;
    int total_consumed=0;
    
    FILE* fpX;
    FILE* fpY;
#ifdef cMPI
    MPI_File mfpX;
    MPI_File mfpY;
#endif
    FILE* tmp_fp;
    
    char filenameX[128]="";
    char filenameY[128]="";
};


void* new_DataGen(DG_Data* DG, const char* filenameX, const char* filenameY, int bs, bool distr_ds, int* dataset_size, int* nbpp,  int method, int num_threads, int buffer_size);
void* start_DataGen(DG_Data* DG);
void* stop_DataGen(DG_Data* DG);
void* get_batch_DataGen(DG_Data* DG, Tensor* in, Tensor* out);  
void* end_DataGen (DG_Data* DG) ;
void* imprime_DG(const char* titulo, DG_Data* DG);
void* imprime_buffer(DG_Data* DG);


// Did not worked
/*
class DataGen {
private:
    Tensor* bufferX[MAX_BUFFER];
    Tensor* bufferY[MAX_BUFFER];
    int n_sizeX=0;
    int n_sizeY=0;
    int* list;
public:
    int dg_id;
    FILE* fpX;
    FILE* fpY;
#ifdef cMPI
    MPI_File mfpX;
    MPI_File mfpY;
#endif
    int buffer_count = 0;
    int ptr_in = 0;
    int ptr_out = 0;
    int ds_ptr = 0;
    sem_t dmutex;
    sem_t llenar;
    sem_t vaciar;
    sem_t imprimir;
    
    pthread_t t[MAX_DG_THREADS];

    int dg_dataset_size;
    int ndimX;
    int ndimY;
    int shape_sizeX;
    int shape_sizeY;
   

    int dg_batch_size = 0;
    int dg_nbpp;
    int dg_buffer_size = 0;
    int dg_num_threads;
   
    bool dg_perfect=true;
  
    DataGen(const string&  filenameX, const string&  filenameY, int bs, bool distr_ds, int* dataset_size, int* nbpp,  bool perfect, int num_threads, int buffer_size);
    void loadXY_perfect(int buffer_index, int ds_ptr, bool perfect);
    void loadXY_perfect_distr(int buffer_index, int ds_ptr, bool perfect);
   
    void* end_data_generator();
    void* start_data_generator(); 
    void* stop_data_generator();
    void* get_batch(Tensor* in, Tensor* out);
    int get_buffer_count();
    int get_nbpp();
    int get_dataset_size();
};

// Did not worked
class DataGen2
{
private:
    int m_numerator {};
    int m_denominator {};
    int dg_id;
   
    int dg_batch_size;
    int dg_num_threads;
   
    int m_dataset_size=0;
    
    
    int ndimX=0;
    int ndimY;
    FILE* fpX;
    FILE* fpY;
#ifdef cMPI
    MPI_File mfpX;
    MPI_File mfpY;
#endif
    int shape_sizeX;
    int shape_sizeY;
    int n_sizeX;
    int n_sizeY;
    Tensor* bufferX[MAX_BUFFER];
    Tensor* bufferY[MAX_BUFFER];
    int* list;
    pthread_t t[MAX_DG_THREADS];
    
public:
    sem_t dmutex;
    sem_t llenar;
    sem_t vaciar;
    sem_t imprimir; 
    int buffer_count = 0;
    int ptr_in = 0;
    int ptr_out = 0;
    int ds_ptr = 0;
    int dg_nbpp;
     int dg_buffer_size;
      bool dg_perfect;
    DataGen2(const string& filenameX, const string& filenameY, int bs, bool distr_ds, int* dataset_size, int* nbpp, bool perfect, int num_threads, int buffer_size);
   
    double getValue() { return static_cast<double>(m_numerator) / m_denominator; }
    void start_data_generator();
    void stop_data_generator();
   //  void* producer_F(void* arg);
    void get_batch(Tensor* in, Tensor* out);
    void loadXY_perfect(int buffer_index, int ds_ptr, bool perfect);
    void loadXY_perfect_distr(int buffer_index, int ds_ptr, bool perfect);
   //  static void * InternalThreadEntryFunc(void * This) {((Fraction *)This)->InternalThreadEntry(); return NULL;} 

 protected:
   // Implement this method in your subclass with the code you want your thread to run. 
  // virtual void InternalThreadEntry();
};
*/


#endif /* DATA_GENERATOR_H */
