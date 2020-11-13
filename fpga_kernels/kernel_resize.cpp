#include <math.h>
#include <stdio.h>
#include <ap_int.h>

#include <hls_stream.h>
#include <iostream> 
#include <queue> 


//#define DEBUG_VERBOSE

extern "C" {

// Fixed parameters (optimized at compilation/synthesis time)

//#define W    6
//#define WOUT 4 //256
//#define H    6
//#define HOUT 4 //256
#define C    8  //canales totales de entrada
#define COUT 8 //canales totales de salida
#define CPI  8 //nuemro de canales por slice con los que trabajamos
#define CPO	 8
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

using namespace std; 

static void read_input(pixel_in_t *ptr, hls::stream<pixel_in_t> &out, int H, int W) {

  pixel_in_t data;
  #pragma HLS ARRAY_PARTITION variable=data dim=0

  printf("read_input");

    //Sending data to padding  in pack of CPI channels
    read_loop_data_load_i:
      for (int r=0; r<H*W*I_ITER; r++) {
      	#pragma HLS PIPELINE II=1
      	pixel_in_t dataR;
      	dataR = ptr[r];

 		printf("r = %d \n", r);
        printf("data.pixel[0] = %6.2f  ", dataR.pixel[0]);
		printf("data.pixel[1] = %6.2f  ", dataR.pixel[1]);
		printf("data.pixel[2] = %6.2f  ", dataR.pixel[2]);
		printf("data.pixel[3] = %6.2f  ", dataR.pixel[3]);
		printf("\n"); 
		printf("data.pixel[4] = %6.2f  ", dataR.pixel[4]);
		printf("data.pixel[5] = %6.2f  ", dataR.pixel[5]);
		printf("data.pixel[6] = %6.2f  ", dataR.pixel[6]);
		printf("data.pixel[7] = %6.2f  ", dataR.pixel[7]);
		printf("\n"); 
      

        out  << dataR;//envio packs de 4 datos a la vez (CPI)
      }
   



#ifdef DEBUG_VERBOSE
  printf("read_input: end\n");
#endif
}

// flip: Flip kernel
//
// Arguments:
//   in: input stream
//   out: output stream
static void downsize(hls::stream<pixel_in_t> &in, hls::stream<pixel_in_t> &out, int HOUT, int WOUT, int H, int W) {

	

	pixel_in_t valor;
	
	printf("downsize\n");
	
	int Wstride = 1 + ((W-1)/ WOUT);
	int Hstride = 1 + ((H-1)/ HOUT);
	int contHOUT = 0;
	int contWOUT = 0;


	int maximoH = 1 + ((H-1)/ 2); //maximo para que la reduccion de la matriz sea exacta y se cojan diferentes puntos salteados de la matriz original
	int maximoW = 1 + ((W-1)/ 2);
	printf("Wstride %d, Hstride %d, maximoW %d maximoH %d\n", Wstride, Hstride, maximoW, maximoH);

	if(maximoH >= HOUT && maximoW >= WOUT){
		//la reduccion es exacta, las matrices o son cuadradas o la reduccion de tamaños son compatibles -> 9x9 a 3x3
		printf("Reduccion de tamaños son compatibles\n");
		write_downsize_loop_exterior1:
		for(int i_iter = 0; i_iter < I_ITER; i_iter++){
			write_downsize_loop_interior_i_1:
			for(int i = 0; i<H; i++){
				write_downsize_loop_interior_j_1:
				for (int j = 0; j <W; j++){
					
					valor = in.read();
					
					if((i % Hstride == 0) || ((i == H-1) && (contHOUT < HOUT))){
						if ((j % Wstride ==0) || ((j == W-1) && (contWOUT < WOUT))){					
							out << valor;
/* 							printf("(%d,%d) contHOUT = %d, contWOUT = %d | ",i,j,contHOUT,contWOUT);
							printf("data.pixel[0] = %6.2f  ",valor.pixel[0]);
							printf("data.pixel[1] = %6.2f  ",valor.pixel[1]);
							printf("data.pixel[2] = %6.2f  ",valor.pixel[2]);
							printf("data.pixel[3] = %6.2f  ",valor.pixel[3]);
							printf("\n");  */
							contWOUT++;
						}
						if(j==W-1){
							contHOUT++;
						}			
					}		
				}
				contWOUT=0;
			}
			contHOUT=0;
		}
	
	}else if(maximoH >= HOUT && maximoW < WOUT){
		//la reduccion de las columnas (W) no es exactas, hay que añadir mas columnas que las que dice el Wstride -> 6x4 a 3x3, Wstride da 2 (3/2), pero se necesitan 3
		int faltan = WOUT-maximoW;
		contHOUT=0;
		contWOUT=0;
		printf("Reduccion de las columnas (W) no es exactas, faltan %d\n", faltan);
		write_downsize_loop_exterior2:
		for(int i_iter = 0; i_iter < I_ITER; i_iter++){
			write_downsize_loop_interior_i_2:
			for(int i = 0; i<H; i++){
				write_downsize_loop_interior_j_2:
				for (int j = 0; j <W; j++){
					
					valor = in.read();
					
					if((i % Hstride == 0) || ((i == H-1) && (contHOUT < HOUT))){
						if ((j % Wstride ==0) || ((j % Wstride !=0) && (contWOUT < faltan))){					
							out << valor;
/* 							printf("(%d,%d) contHOUT = %d, contWOUT = %d | ",i,j,contHOUT,contWOUT);
							printf("data.pixel[0] = %6.2f  ",valor.pixel[0]);
							printf("data.pixel[1] = %6.2f  ",valor.pixel[1]);
							printf("data.pixel[2] = %6.2f  ",valor.pixel[2]);
							printf("data.pixel[3] = %6.2f  ",valor.pixel[3]);
							printf("\n");  */
							if(j % Wstride !=0){
								contWOUT++;
							}
						}
						if(j==W-1){
							contHOUT++;
						}				
					}
				}
				contWOUT=0;
			}
			contHOUT=0;
		}
		
	}else if(maximoH < HOUT && maximoW >= WOUT){
		//la reduccion de las filas (H) no es exactas, hay que añadir mas filas que las que dice el Wstride -> 7x4 a 5x2, 
		//Hstride da 2 (7/2) y se seleccionas 4 filas de la original, cuando se necesitan 5
		int faltan = HOUT-maximoH;
		contHOUT=0;
		contWOUT=0;
		printf("Reduccion de las filas (H) no es exactas, faltan %d\n", faltan);
		write_downsize_loop_exterior3:
		for(int i_iter = 0; i_iter < I_ITER; i_iter++){
			write_downsize_loop_interior_i_3:
			for(int i = 0; i<H; i++){
				write_downsize_loop_interior_j_3:
				for (int j = 0; j <W; j++){
					
					valor = in.read();
					
					if((i % Hstride == 0) || ((i % Hstride != 0) && (contHOUT < faltan))){
						if ((j % Wstride ==0) || ((j == W-1) && (contWOUT < WOUT))){					
							out << valor;
/* 							printf("(%d,%d) contHOUT = %d, contWOUT = %d | ",i,j,contHOUT,contWOUT);
							printf("data.pixel[0] = %6.2f  ",valor.pixel[0]);
							printf("data.pixel[1] = %6.2f  ",valor.pixel[1]);
							printf("data.pixel[2] = %6.2f  ",valor.pixel[2]);
							printf("data.pixel[3] = %6.2f  ",valor.pixel[3]);
							printf("\n");  */
							contWOUT++;
						}
						if(j==W-1 && (i % Hstride != 0)){
							contHOUT++;
						}				
					}
				}
				contWOUT=0;
			}
			contHOUT=0;
		}
		
	}else{
		//tanto la reduccion de las columnas (W) como las filas (H) no es exacta -> 7x4 a 5x3, hay que añadir en ambos lados
		int faltanW = WOUT-maximoW;
		int faltanH = HOUT-maximoH;
		contHOUT=0;
		contWOUT=0;
		printf("Reduccion de las filas (H) y columnas (W) no es exactas, faltan %d filas y %d columnas\n ", faltanH, faltanW);
		
		write_downsize_loop_exterior4:
		for(int i_iter = 0; i_iter < I_ITER; i_iter++){
			write_downsize_loop_interior_i_4:
			for(int i = 0; i<H; i++){
				write_downsize_loop_interior_j_4:
				for (int j = 0; j <W; j++){
					
					valor = in.read();
					
					if((i % Hstride == 0) || ((i % Hstride != 0) && (contHOUT < faltanH))){
						if ((j % Wstride ==0) || ((j % Wstride !=0) && (contWOUT < faltanW))){					
							out << valor;
/* 							printf("(%d,%d) contHOUT = %d, contWOUT = %d | ",i,j,contHOUT,contWOUT);
							printf("data.pixel[0] = %6.2f  ",valor.pixel[0]);
							printf("data.pixel[1] = %6.2f  ",valor.pixel[1]);
							printf("data.pixel[2] = %6.2f  ",valor.pixel[2]);
							printf("data.pixel[3] = %6.2f  ",valor.pixel[3]);
							printf("\n");  */
							if(j % Wstride !=0){
								contWOUT++;
							}
						}
						if(j==W-1 && (i % Hstride != 0)){
							contHOUT++;
						}					
					}
				}
				contWOUT=0;
			}
			contHOUT=0;
		}
	}
}


static void upsize(hls::stream<pixel_in_t> &in, hls::stream<pixel_in_t> &out, int HOUT, int WOUT, int H, int W) {

	

	pixel_in_t valor;
	
     pixel_in_t repetidos[256];
	#pragma HLS ARRAY_PARTITION variable=repetidos dim=0
	printf("upsize ");
	
	int Wrepeat = WOUT/W; //las columnas
	int Hrepeat = HOUT/H; //las filas
	
	int countW = 0;
	int countH = 0;
	int countHaux = 0;
	
	int enableW = 1;
	int enableH = 0;
	int enableHfalta = 0;
	int contReptH = 0;
	int contReptW = 0;
	
	int faltanH = HOUT - (Hrepeat * H);
	int faltanW = WOUT - (Wrepeat * W);
	printf("Wrepeat %d, Hrepeat %d, faltanW %d faltanH %d\n", Wrepeat, Hrepeat, faltanW, faltanH);

	write_upsize_loop_exterior:
	for(int i_iter = 0; i_iter < I_ITER; i_iter++){
		write_upsize_loop_interior:
		for(int i = 0; i<HOUT; i++){
			for (int j = 0; j <WOUT; j++){

				if(enableH){
					//repetimos fila
					out << repetidos[j];
					printf("enableH (%d,%d) contReptH = %d countH = %d",i,j,contReptH, countH);
					printf("\n"); 
					if(j == WOUT - 1){
						enableH = 0;
					}	
				}else if(enableHfalta){	
					//repetimos la que sobra
					out << repetidos[j];
					printf("enableH FALTA (%d,%d) contReptH = %d countH = %d",i,j,contReptH, countH);
					printf("\n"); 
					if(j == WOUT - 1){
						enableHfalta = 0;
						countH++;
					}
				}else{				
					//leemos nueva fila
					if(enableW){
						printf("enableW 0 lee (%d,%d)\n",i,j);
						valor = in.read();
						enableW = 0;
					}
					//comprobamos si hay repetidos
					if(contReptW < Wrepeat){
						out << valor;
						printf("enableW 1 (%d,%d) contReptW = %d countW = %d",i,j, contReptW, countW);
						printf("\n"); 
						repetidos[j]=valor;
						contReptW++;
						if(j == WOUT - 1){
							enableW = 1;
							contReptW = 0;
							countW = 0;
						}else if(contReptW == Wrepeat && countW == faltanW){
							enableW = 1;
							contReptW = 0;
						}	
					}else if(contReptW == Wrepeat){
						//comprobamos si faltan (division inexacta)
						if(countW < faltanW){
							out << valor;
							printf("enableW 2(%d,%d) contReptW = %d countW = %d",i,j, contReptW, countW);
							printf("\n"); 
							repetidos[j]=valor;
							countW++;
							enableW = 1;
							contReptW = 0;
						}else{
							//contReptW == Wrepeat y no faltan, reiniciamos.
							printf("enableW 4 (%d,%d) contReptW = %d countW = %d\n",i,j, contReptW, countW);
							contReptW = 0;
							enableW = 1;		
							countW = 0;
						}
					}else if(countW < faltanW){
						//se acaba la fila
						out << valor;
						printf("enableW 3(%d,%d) countW = %d",i,j, countW);
						printf("\n"); 
						countW = 0;
					}
				}
			}
			contReptH++;
			if(!enableH && (contReptH < Hrepeat)){
				printf("i = %d contReptH = %d countH = %d\n", i, contReptH, countH);
				enableH = 1;
			}else if(contReptH == Hrepeat && countH < faltanH && countH == countHaux){
				printf("FALTAN H i = %d contReptH = %d countH = %d countHaux= %d\n", i, contReptH, countH, countHaux);
				enableHfalta = 1;
			}else{
				printf("FINAL H i = %d contReptH = %d countH = %d\n", i, contReptH, countH);
				enableHfalta = 0;
				contReptH = 0;
				enableH = 0;
				countHaux = countH;
			}
		}
		countH = 0;
		contReptH = 0;
		enableH = 0;
		countHaux = 0;
	}
}

// --------------------------------------------------------------------------------

static void write_output(pixel_in_t *ptr, hls::stream<pixel_in_t> &in, int HOUT, int WOUT, int H, int W) {
  
    printf("entraaaa444444444\n");
write_output_o_iter_loop:
  for (int o_iter = 0; o_iter<O_ITER; o_iter++){
    // writes must be performed with pixel_in_t struct

    write_output_data_size_loop:
    for (int i=0; i<HOUT*WOUT; i++) {
      int addr_d = i*O_ITER + o_iter;
      ptr[addr_d] = in.read();

      printf("o_iter = %d  i = %d \n", o_iter, i);
      printf("ptr.pixel[0] = %6.2f ", ptr[addr_d].pixel[0]);
      printf("ptr.pixel[1] = %6.2f ", ptr[addr_d].pixel[1]);
      printf("ptr.pixel[2] = %6.2f ", ptr[addr_d].pixel[2]);
      printf("ptr.pixel[3] = %6.2f ", ptr[addr_d].pixel[3]);
	  printf("\n"); 
	  printf("ptr.pixel[4] = %6.2f ", ptr[addr_d].pixel[4]);
      printf("ptr.pixel[5] = %6.2f ", ptr[addr_d].pixel[5]);
      printf("ptr.pixel[6] = %6.2f ", ptr[addr_d].pixel[6]);
      printf("ptr.pixel[7] = %6.2f ", ptr[addr_d].pixel[7]);
	  printf("\n"); 

    }
  }
  
/*   	pixel_in_t dataW;
	#pragma HLS ARRAY_PARTITION variable=dataW dim=0
	 

	write_output_data_upsize_size_loop:
    for (int i=0; i<HOUT*WOUT*I_ITER; i++) {
	    dataW = in.read();
	    ptr[i] = dataW;
		printf("i = %d \n", i);
		printf("data.pixel[0] = %6.2f  ", dataW.pixel[0]);
		printf("data.pixel[1] = %6.2f  ", dataW.pixel[1]);
		printf("data.pixel[2] = %6.2f  ", dataW.pixel[2]);
		printf("data.pixel[3] = %6.2f  ", dataW.pixel[3]);
		printf("\n");  
    } */

	 

  
  printf("entraaaa55555555555\n");

}

static void selector(hls::stream<pixel_in_t> &in, hls::stream<pixel_in_t> &out, int HOUT, int WOUT, int H, int W, int mode){
  if(mode == 0){
	printf("entraaaa11111111111111\n");
	upsize(in, out, HOUT, WOUT, H, W);
  }
  else{
	printf("entraaaa2222222222222222\n");
	downsize(in, out, HOUT, WOUT, H, W);
  }
}

void k_resize(pixel_in_t *ptr_data, pixel_in_t *ptr_out, int HOUT, int WOUT, int H, int W, int mode) {

  #pragma HLS INTERFACE m_axi port=ptr_data offset=slave bundle=gmem   max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE m_axi port=ptr_out  offset=slave bundle=gmem   max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE s_axilite port=HOUT  bundle=control
  #pragma HLS INTERFACE s_axilite port=WOUT  bundle=control
  #pragma HLS INTERFACE s_axilite port=H  bundle=control
  #pragma HLS INTERFACE s_axilite port=W  bundle=control
  #pragma HLS INTERFACE s_axilite port=mode  bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control


  // ptr_data struct to be packed as a single element vector (to improve memory read)
  // the compiler will do full structure access (all elements of structure)
  #pragma HLS data_pack variable = ptr_data
  #pragma HLS data_pack variable = ptr_out

  // input and output streams
  static hls::stream<pixel_in_t> out_read;
  static hls::stream<pixel_in_t> out_resize;

  // stream sizes
  #pragma HLS STREAM variable = out_read depth = 32
  #pragma HLS STREAM variable = out_resize depth = 32

  
  printf("De (%d,%d) a (%d,%d) mode %d\n", H, W, HOUT,WOUT,mode);

  #pragma HLS dataflow
  read_input(ptr_data, out_read, H, W);
  selector(out_read, out_resize, HOUT, WOUT, H, W, mode);
  write_output(ptr_out, out_resize, HOUT, WOUT, H, W);
  
  printf("entraaaa3333333333333333333\n");
}

} // end extern "C"
