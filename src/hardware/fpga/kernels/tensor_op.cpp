/**********

*******************************************************************************/
#include <stdio.h>
#include <math.h>
#define BUFFER_SIZE 1024
static int rnd_seed = 123132545;

void set(float *tensor, float fp, int tam){
#pragma HLS INLINE 
//    float tensor_buffer[BUFFER_SIZE];    // Local memory to store input tensor 
    //float tensor_out_buffer[BUFFER_SIZE];    // Local memory to store tensor results
     //Per iteration of this loop perform BUFFER_SIZE vector addition
    for(int i = 0; i < tam;  i++)
    {
        #pragma HLS PIPELINE II=1
            //perform vector addition
            tensor[i] = fp; 
    }
    printf("SET %f\n", fp);
}

void mult(float *tensor, float fp, int tam){
#pragma HLS INLINE 
    float tensor_buffer[BUFFER_SIZE];    // Local memory to store input tensor 
     //Per iteration of this loop perform BUFFER_SIZE vector addition
    for(int i = 0; i < tam;  i += BUFFER_SIZE)
    {

        int chunk_size = BUFFER_SIZE;
        //boundary checks
        if ((i + BUFFER_SIZE) > tam) 
            chunk_size = tam - i;

        // Transferring data in bursts hides the memory access latency as well as improves bandwidth utilization and efficiency of the memory controller.
        // It is recommended to infer burst transfers from successive requests of data from consecutive address locations.
        // A local memory vl_local is used for buffering the data from a single burst. The entire input vector is read in multiple bursts.
        // The choice of LOCAL_MEM_SIZE depends on the specific applications and available on-chip memory on target FPGA. 
        // burst read of v1 and v2 vector from global memory
        read: for (int j = 0 ; j < chunk_size ; j++){
            tensor_buffer[j] = tensor[i + j];
        }

        // PIPELINE pragma reduces the initiation interval for loop by allowing the
        // concurrent executions of operations
        tensor_mult: for (int j = 0 ; j < chunk_size; j ++){
        #pragma HLS PIPELINE II=1
            //perform vector operation
            tensor[j] = tensor_buffer[j] * fp; 
        }
    }

}

void sum(float *tensor, float fp, int tam) {
#pragma HLS INLINE 
    float tensor_buffer[BUFFER_SIZE];    // Local memory to store input tensor 
     //Per iteration of this loop perform BUFFER_SIZE vector addition
    for(int i = 0; i < tam;  i += BUFFER_SIZE)
    {
        int chunk_size = BUFFER_SIZE;
        //boundary checks
        if ((i + BUFFER_SIZE) > tam) 
            chunk_size = tam - i;

        read: for (int j = 0 ; j < chunk_size ; j++){
            tensor_buffer[j] = tensor[i + j];
        }

        tensor_add: for (int j = 0 ; j < chunk_size; j ++){
        #pragma HLS PIPELINE II=1
            //perform vector operation
            tensor[j] = tensor_buffer[j] + fp; 
        }
        //burst write the result
    }
}

void set_log(float *tensor, int tam){
#pragma HLS INLINE 
    float tensor_buffer[BUFFER_SIZE];    // Local memory to store input tensor 
    for(int i = 0; i < tam;  i += BUFFER_SIZE)
    {
        int chunk_size = BUFFER_SIZE;
        //boundary checks
        if ((i + BUFFER_SIZE) > tam) 
            chunk_size = tam - i;

        read: for (int j = 0 ; j < chunk_size ; j++){
            tensor_buffer[j] = tensor[i + j];
        }

        tensor_log: for (int j = 0 ; j < chunk_size; j ++){
        #pragma HLS PIPELINE II=1
            //perform vector addition
            tensor[j] = log(tensor_buffer[j]); 
        }
    }
}
void set_exp(float *tensor, int tam){
#pragma HLS INLINE 
    float tensor_buffer[BUFFER_SIZE];    // Local memory to store input tensor 
    for(int i = 0; i < tam;  i += BUFFER_SIZE)
    {
        int chunk_size = BUFFER_SIZE;
        //boundary checks
        if ((i + BUFFER_SIZE) > tam) 
            chunk_size = tam - i;

        read: for (int j = 0 ; j < chunk_size ; j++){
            tensor_buffer[j] = tensor[i + j];
        }

        tensor_exp: for (int j = 0 ; j < chunk_size; j ++){
        #pragma HLS PIPELINE II=1
            //perform vector addition
            tensor[j] = exp(tensor_buffer[j]); 
        }
    }
}
void set_sqrt(float *tensor, int tam){
#pragma HLS INLINE
    for (int i = 0 ; i < tam; i ++){
 //   #pragma HLS PIPELINE II=1
       tensor[i] = sqrt(tensor[i]);
    }

}
void set_sqr(float *tensor, int tam){
#pragma HLS INLINE 
    tensor_pow: for (int i = 0 ; i < tam; i ++){
    #pragma HLS PIPELINE II=1
       tensor[i] = tensor[i]*tensor[i];
    }
}
float total_sum(float *tensor, int tam){
#pragma HLS INLINE 
     float tensor_buffer[BUFFER_SIZE];
     float acc = 0;
     for(int i = 0; i < tam;  i += BUFFER_SIZE)
     {
        int chunk_size = BUFFER_SIZE;
        //boundary checks
        if ((i + BUFFER_SIZE) > tam)
            chunk_size = tam - i;

        read: for (int j = 0 ; j < chunk_size ; j++){
            tensor_buffer[j] = tensor[i + j];
        }
        tensor_totalsum: for (int j = 0 ; j < chunk_size; j ++){
        #pragma HLS PIPELINE II=1
            //perform vector addition
            acc +=tensor_buffer[i];
        }
     }
     return acc;
}
float total_abs(const float *tensor, int tam){
#pragma HLS INLINE 
     float tensor_buffer[BUFFER_SIZE];
     float acc = 0;
     for(int i = 0; i < tam;  i += BUFFER_SIZE)
     {
        int chunk_size = BUFFER_SIZE;
        //boundary checks
        if ((i + BUFFER_SIZE) > tam)
            chunk_size = tam - i;

        read: for (int j = 0 ; j < chunk_size ; j++){
            tensor_buffer[j] = tensor[i + j];
        }
        tensor_log: for (int j = 0 ; j < chunk_size; j ++){
        #pragma HLS PIPELINE II=1
            //perform vector addition
            acc +=fabs(tensor_buffer[i]);
        }
     }
     return acc;
}

int rand_int_function(){
#pragma HLS INLINE
{
    int k1;
    int ix = rnd_seed;
	
    k1 = ix / 127773;
    ix = 16807 * (ix - k1 * 127773) - k1 * 2836;
    if (ix < 0)
        ix += 2147483647;
    rnd_seed = ix;
    return rnd_seed;
}

}


void rand_gaussian(float *tensor, float fp, int tam){
#pragma HLS INLINE
     for (int i=0; i<tam; i++){
        #pragma HLS PIPELINE II=1
	tensor[i]=0.333;//fp*rand_int_function();
     }
}

extern "C" {
void tensor_op( 
        float *tensor, // Output Tensor
        float fp,       // float parameter
        int tam,         // Tensor total elements
	int kernel_id 
        )
{

#pragma HLS INTERFACE m_axi port=tensor offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=tensor  bundle=control
#pragma HLS INTERFACE s_axilite port=fp bundle=control
#pragma HLS INTERFACE s_axilite port=tam bundle=control
#pragma HLS INTERFACE s_axilite port=kernel_id bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
   float acc; 
   switch (kernel_id) {
       case 0: mult(tensor, fp, tam); break;
       case 1: sum(tensor, fp, tam); break;
       case 2: set_log(tensor, tam); break;
       case 3: set_exp(tensor, tam); break;
       case 4: set_sqrt(tensor, tam); break;
       case 5: set_sqr(tensor, tam); break;
       case 6: acc = total_sum(tensor, tam); break;
       case 7: acc = total_abs(tensor, tam); break;
       case 8: set(tensor, fp, tam); break;
       case 11: rand_gaussian(tensor, fp, tam); break;
   /*  case 8: rand_uniform(); break;
       case 9: rand_suniform(); break;
       case 10: rand_gaussian(); break;
       case 11: rand_binary(); break;   */
   }
}

} 
