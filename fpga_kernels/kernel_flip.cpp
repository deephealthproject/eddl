#include <math.h>
#include <stdio.h>
#include <ap_int.h>

#include <hls_stream.h>

//#define DEBUG_VERBOSE

extern "C" {

// Fixed parameters (optimized at compilation/synthesis time)

#define W    256 // input width
#define H    256// input height
#define C    8  //canales totales de entrada
#define COUT 8 //canales totales de salida
#define CPI  4 //nuemro de canales por slice con los que trabajamos
#define CPO	 4
#define I_ITER 	 C/CPI //tamaño del grupo, nº canales entre tamaño slice
#define O_ITER   COUT/CPO // itreciones por entrada

#define LOAD_MODEL
#define READ_MODEL
#define READ_INPUT
#define WRITE_OUTPUT

// pixel_in
struct pixel_in_t {
  float pixel[CPI]; //pixel de 4 datos
};



static void read_input(pixel_in_t *ptr, hls::stream<pixel_in_t> &out) {

  pixel_in_t dataR;
  #pragma HLS ARRAY_PARTITION variable=dataR dim=0

  

    //Sending data to padding  in pack of CPI channels
    read_loop_data_load_i:
      for (int r=0; r<H*W*I_ITER; r++) {
      	#pragma HLS PIPELINE II=1
      	pixel_in_t dataR;
      	dataR = ptr[r];

/*  		printf("r = %d \n", r);
        printf("data.pixel[0] = %6.2f  ", dataR.pixel[0]);
		printf("data.pixel[1] = %6.2f  ", dataR.pixel[1]);
		printf("data.pixel[2] = %6.2f  ", dataR.pixel[2]);
		printf("data.pixel[3] = %6.2f  ", dataR.pixel[3]);
		printf("\n");  */
      

        out  << dataR;//envio packs de 4 datos a la vez (CPI)
      }
	 // printf("\nSegunda tanda \n");
	

   



#ifdef DEBUG_VERBOSE
  printf("read_input: end\n");
#endif
}

// flip: Flip kernel
//
// Arguments:
//   in: input stream
//   out: output stream
/* static void flip(hls::stream<pixel_in_t> &in, hls::stream<pixel_in_t> &out) {

	pixel_in_t buffer [2][H*W];//buffers de 4 canales, y son H*W pixels C/CPI 
	#pragma HLS ARRAY_PARTITION variable=buffer cyclic factor=2 dim=0
	
  printf("entraaaa");
	//codigo del flip
	int indice = 0;
	int primero = H*W - W;
	int entry_read = 0; 
	int entry_write  = 1;
	
	for(int i = 0; i<C/CPI; i++){
		for (int j = 0; j < H*W; j++){
			
			buffer[entry_read][j]= in.read();
			
			if(i%2 != 0){//si no es par ni 0, ya se ha leido 1 tanda de 4 canales entera -> se puede escribir
				
				out << buffer[entry_write][primero + indice];
				
				if(indice != (W-1)){//el indice va sumando hasta la anchuta total de la imagen
					indice++;
				}else{//se reinicia el indice y el primero pasa a tener el valor del primero de una fila inferior
					indice = 0;
					primero = primero - W;
				}
			}
		}
		entry_read = (entry_read + 1) % 2;
		entry_write = (entry_write + 1) %2;
	}
	
	//falta la ultima iteracion, se necesita 1 bucle mas porque la primera lectura no se ha escrito
	indice = 0;
	primero = H*W - W;
	for (int p = 0; p < H*W; p++){
		out << buffer[entry_write][primero + indice];
		
		if(indice != (W-1)){//el indice va sumando hasta la anchuta total de la imagen
			indice++;
		}else{//se reinicia el indice y el primero pasa a tener el valor del primero de una fila inferior
			indice = 0;
			primero = primero - W;
		}
		
	}

} */

static void flip(hls::stream<pixel_in_t> &in, hls::stream<pixel_in_t> &out) {

	pixel_in_t buffer [2][H*W];//buffers de 4 canales, y son H*W pixels C/CPI 
	#pragma HLS ARRAY_PARTITION variable=buffer cyclic factor=2 dim=0
	
  printf("entraaaa");
	//codigo del flip
	int ultimo = H*W-1;
	int entry_read = 0; 
	int entry_write  = 1;
	
	for(int i = 0; i<C/CPI; i++){
		for (int j = 0; j < H*W; j++){
			
			buffer[entry_read][j]= in.read();
			
			if(i%2 != 0){//si no es par ni 0, ya se ha leido 1 tanda de 4 canales entera -> se puede escribir
				
				out << buffer[entry_write][ultimo];
				
				if(ultimo != 0){//el indice va sumando hasta la anchuta total de la imagen
					ultimo--;
				}else{//se reinicia el indice y el primero pasa a tener el valor del primero de una fila inferior
					ultimo = H*W-1;
				}
			}
		}
		entry_read = (entry_read + 1) % 2;
		entry_write = (entry_write + 1) %2;
	}
	
	//falta la ultima iteracion, se necesita 1 bucle mas porque la primera lectura no se ha escrito
	for (int p = H*W-1; p >= 0; p--){
		out << buffer[entry_write][p];	
	}

}

// --------------------------------------------------------------------------------

static void write_output(pixel_in_t *ptr, hls::stream<pixel_in_t> &in) {
  
/*   write_output_o_iter_loop:
  for (int o_iter = 0; o_iter<O_ITER; o_iter++){
    write_output_data_size_loop:
    for (int i=0; i<H*W; i++) {
      int addr_d = i*O_ITER + o_iter;
      ptr[addr_d] = in.read();

    }
  } */
	pixel_in_t dataW;
	#pragma HLS ARRAY_PARTITION variable=dataW dim=0
	 
	 
    write_output_data_size_loop:
    for (int i=0; i<H*W*I_ITER; i++) {
	    dataW = in.read();
	    ptr[i] = dataW;
/* 		printf("i = %d \n", i);
		printf("data.pixel[0] = %6.2f  ", dataW.pixel[0]);
		printf("data.pixel[1] = %6.2f  ", dataW.pixel[1]);
		printf("data.pixel[2] = %6.2f  ", dataW.pixel[2]);
		printf("data.pixel[3] = %6.2f  ", dataW.pixel[3]);
		printf("\n");  */
    }


}

void k_flip(pixel_in_t *ptr_data, pixel_in_t *ptr_out) {

  //#pragma HLS INTERFACE s_axilite port=W bundle=control
  //#pragma HLS INTERFACE s_axilite port=H bundle=control
  #pragma HLS INTERFACE m_axi port=ptr_data offset=slave bundle=gmem   max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE m_axi port=ptr_out  offset=slave bundle=gmem   max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE s_axilite port=return bundle=control

  // ptr_data struct to be packed as a single element vector (to improve memory read)
  // the compiler will do full structure access (all elements of structure)
  #pragma HLS data_pack variable = ptr_data
  #pragma HLS data_pack variable = ptr_out

  // input and output streams
  static hls::stream<pixel_in_t> out_read;
  static hls::stream<pixel_in_t> out_flip;

  // stream sizes
  #pragma HLS STREAM variable = out_read depth = 32
  #pragma HLS STREAM variable = out_flip depth = 32

  #pragma HLS dataflow
  read_input(ptr_data, out_read);
  flip(out_read, out_flip);
  write_output(ptr_out, out_flip);
}

} // end extern "C"
