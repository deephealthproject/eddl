#include "conv2D.h"

#include <hls_stream.h>

// ---------------------------------------------------------------------------------------
// read_bias. Reading bias from memory and sending to conv module
//
// Arguments:
//   b_ptr               : pointer to bias
//   offset_bias         : offset to bias
//   b_out               : output stream
//
// All the bias are read and sent through the out stream
//
static void read_bias(int offset_bias, pixel_out_t *b_ptr, hls::stream<pixel_out_t> &out) {

  #ifdef DEBUG_READ_BIAS
  printf("READ_BIAS: start\n");
  #endif

  pixel_out_t bias;
  #pragma HLS ARRAY_PARTITION variable=bias complete dim=0

  bias = b_ptr[offset_bias];
  out << bias;

  #ifdef DEBUG_READ_BIAS
  printf("READ_BIAS: offset_bias = %d\n", offset_bias);
  printf("READ_BIAS: bias = ");
  for (int c=0; c<CPO; c++) printf(" %f ", float(bias.pixel[c]));
  printf("\n");
  #endif

  #ifdef DEBUG_READ_BIAS
  printf("READ_BIAS: end\n");
  #endif
}

// ---------------------------------------------------------------------------------------
// read_kernel. Reads kernels and sends them through the stream
//
// Arguments:
//   I_ITER              : Number of input iterations (I / CPI)
//   k_ptr               : pointer to kernels
//   offset_kernel       : offset to kernels
//   k_out               : output stream
//
// kernels are stored in memory with the format GO x GI x CPO x CPI x KH x KW
// This storage formats lets the module to read memory sequentially and send all the
// kernels in the same order they are read through the output stream.
// kernels are sent in frame structures (3x3 grid)
//
static void read_kernel(int I_ITER, int offset_kernel, data_type *k_ptr, hls::stream<kernel_t> &k_out){

  #ifdef DEBUG_READ_KERNEL
  printf("READ_KERNEL: start\n");
  #endif

  kernel_t k;
  int cnt = 0;
  #pragma HLS array_partition variable=k complete dim=0

  for (int i=0; i<I_ITER; i++) {
	  DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=I_REFERENCE/CPI)
	  for (int cpo=0; cpo < CPO; cpo++) {
		  for (int cpi=0; cpi < CPI; cpi++) {
			  for (int p=0; p<9; p++) {
				  #pragma HLS pipeline II=1
				  k.pixel[cpo][cpi][p] = k_ptr[offset_kernel+cnt];
				  cnt = cnt + 1;
			  }
		  }
	  }
	  k_out << k;
	  #ifdef DEBUG_READ_KERNEL
	  for (int cpo=0; cpo<CPO; cpo++) {
		  for (int cpi=0; cpi<CPI; cpi++) {
		      printf("READ_KERNEL: Kernel read for cpi=%d cpo=%d : ", cpi, cpo);
		      for (int p=0;p<9; p++) printf(" %6.4f ", k.pixel[cpo][cpi][p]);
		      printf("\n");
		  }
	  }
	  #endif
  }

  #ifdef DEBUG_READ_KERNEL
  printf("READ_KERNEL: end\n");
  #endif
}

// ---------------------------------------------------------------------------------------
// read_data_channels. Reads all data channels and send it through the output streams
//
// Arguments:
//   H, W                : Data channel height and width
//   rows                : Number of rows of the frame to read
//   num_extra_rows      : Number of extra rows to read
//   I_ITER              : Number of input iterations (I / CPI)
//   ptr                 : pointer to input data
//   offset              : offsets within input data for each channel
//   out                 : output streams for each channel
//   enable_read_channel : enables for each channel. If not set the module produces just zeros and does not read memory
//
static void read_data_channels(int H, int W, int rows, int I_ITER, ap_uint<512> *ptr, int offset, int num_extra_rows, int channel_blocks, hls::stream<read_block_t> out[CPI], int *enable_read_channel) {

  #ifdef DEBUG_READ_DATA
  printf("READ_DATA: starts\n");
  #endif

  int num_pixels = (num_extra_rows + rows) * W;
  int channel_size = H * W;
  read_block_t bx[CPI];
  DO_PRAGMA(HLS ARRAY_PARTITION variable=bx complete dim=0)

    read_data_channels_loop_I_ITER:
    for (int i_iter = 0; i_iter < I_ITER; i_iter++) {
      DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=I_REFERENCE/CPI)

      // each channel has its first block
      int offset_[CPI];
      int first_block_[CPI];
      // We read in chunks of CHUNK_SIZE blocks
      int channel_blocks_remaining_[CPI];
      read_data_channels_loop_CPI_init:
      for(int i = 0; i<CPI; i++){
        #pragma HLS UNROLL
        offset_[i] = offset + (channel_size * CPI * i_iter) + (channel_size * i);
        first_block_[i] = offset_[i] / READ_BLOCK_SIZE;
        channel_blocks_remaining_[i] = channel_blocks;
        #ifdef DEBUG_READ_DATA
        printf("READ_DATA: cpi %d -> offset %d first_block %d remaining %d\n", i, offset_[i], first_block_[i], channel_blocks_remaining_[i]);
        #endif
      }

      read_data_channels_loop_CHUNKS:
      for (int block = 0; block < channel_blocks; block++) {
    	DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=WHMAX/READ_BLOCK_SIZE)
    	read_data_channels_loop_CPI:
        for(int i = 0; i < CPI; i++){
          #pragma HLS pipeline
          ap_uint<512> data_read;
          if (enable_read_channel[i]) {
        	  data_read = ptr[first_block_[i]];
              read_data_channels_loop_block_pixels:
              for (int p=0; p<READ_BLOCK_SIZE; p++) {
                DO_PRAGMA(HLS UNROLL)
                int first = p * DATA_TYPE_WIDTH;
                int last = first + DATA_TYPE_WIDTH-1;
                unsigned int tmp = data_read.range(last, first);
                data_type datum = *(data_type*)(&tmp);
                bx[i].pixel[p] = datum;
              }
              if (enable_read_channel[i] && channel_blocks_remaining_[i]) out[i] << bx[i];
              if (channel_blocks_remaining_[i]) channel_blocks_remaining_[i] = channel_blocks_remaining_[i] - 1;
              first_block_[i] = first_block_[i] + 1;
          }
        }
      }
    } //i_iter

  #ifdef DEBUG_READ_DATA
  printf("READ_DATA: ends\n");
  #endif

}

// -------------------------------------------------------------------
static void serialize_and_filter(int I_ITER, int num_pixels, int channel_blocks, int channel_size, int offset,
		                         hls::stream<read_block_t> &in, hls::stream<data_type> &out, int enable) {

  #ifdef DEBUG_SERIALIZE
  printf("SERIALIZE: starts (num_pixels = %d)\n", num_pixels);
  #endif

  int num_pixels_cnt;

  // Zero block initialization
  read_block_t data_zeros;
  for (int b=0; b<READ_BLOCK_SIZE; b++) {
    #pragma HLS UNROLL
    data_zeros.pixel[b] = 0;
  }

  int iters = I_ITER * channel_blocks * READ_BLOCK_SIZE;
  int b = 0;
  int p = 0;
  int iter = 0;
  int offset_ch = 0;
  for (int i_iter=0; i_iter < iters; i_iter++) {
	DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=I_REFERENCE/CPI * READ_BLOCK_SIZE * (WHMAX / READ_BLOCK_SIZE))
    #pragma HLS pipeline II=1
    // offset
    if ((b==0) && (p==0)) {
      offset_ch = (offset + (channel_size * CPI * iter)) % READ_BLOCK_SIZE;
      num_pixels_cnt = num_pixels;
    }
    read_block_t bx;
    DO_PRAGMA(HLS ARRAY_PARTITION variable=bx dim=0 complete)
    if (p==0) {
      if (enable) bx = in.read(); else bx = data_zeros;
    }
    if ((offset_ch==0) && (num_pixels_cnt !=0)) {
      out << bx.pixel[p];
      num_pixels_cnt = num_pixels_cnt - 1;
      #ifdef DEBUG_SERIALIZE
      printf("SERIALIZE: pixel forwarded %f\n", (float)bx.pixel[p]);
      #endif
    } else {
      offset_ch = offset_ch - 1;
    }
    p = p + 1;
    if (p == READ_BLOCK_SIZE) {
      p = 0;
      b = b + 1;
      if (b == channel_blocks) {
        b = 0;
        iter = iter + 1;
      }
    }
  }

  #ifdef DEBUG_SERIALIZE
  printf("SERIALIZE: ends (remaining pixels to send %d)\n", num_pixels_cnt);
  #endif

}

template <int LEVELS>
static void ch_serialize_and_filter(int I_ITER, int num_pixels, int channel_blocks, int channel_size,
                                                                int *offset_read_data_channel_i,
                                                                hls::stream<read_block_t> stream_data_ch_0[LEVELS],
                                                                hls::stream<data_type> stream_data_ch_1[LEVELS],
                                                                int *enable_read){

#pragma HLS inline
ch_serialize_and_filter:
  for (int i = 0; i < LEVELS; i++) {
    #pragma HLS UNROLL
    serialize_and_filter(I_ITER, num_pixels, channel_blocks, channel_size, offset_read_data_channel_i[i], stream_data_ch_0[i], stream_data_ch_1[i], enable_read[i]);
  }
}

// ---------------------------------------------------------------------------------------
// join. Joins input streams of pixels and combines them to produce groups of pixels
//
// Arguments:
//   H, W                : Data channel height and width
//   I_ITER              : Number of input iterations (I / CPI)
//   in                  : input streams
//   out                 : output stream
//
// The input streams have width of BLOCK_SIZE elements whereas the output stream
// has width of CPI elements. This module gets the first elements of all input
// streams and produces an output data, then it takes the second elements of all
// input streams and produces a new output data, and so on... For every received
// input data from all streams the join module uses BLOCK_SIZE cycles to produce
// BLOCK_SIZE data items. All data items are sent through the output stream
//
static void join(int H, int W, int I_ITER, int num_extra_rows, hls::stream<data_type> in[CPI], hls::stream<pixel_in_t> &out) {

  #ifdef DEBUG_JOIN
  printf("JOIN: starts\n");
  #endif

  int num_pixels = (H + num_extra_rows) * W;                    // pixels to read

  #ifdef DEBUG_JOIN
  printf("JOIN: Expected pixels = %d\n", num_pixels);
  #endif

  for (int i_iter = 0; i_iter < I_ITER; i_iter++) {
	DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=I_REFERENCE/CPI)

    join_loop:
    for (int r=0; r<num_pixels; r++) {
      DO_PRAGMA(HLS loop_tripcount  min=1 max=WHMAX)
      #pragma HLS PIPELINE II=1
      pixel_in_t data;
      DO_PRAGMA(HLS ARRAY_PARTITION variable=data complete dim=0)
      for(int i=0; i<CPI; i++){
        DO_PRAGMA(HLS loop_tripcount  min=1 max=CPI)
        #pragma HLS UNROLL
        data.pixel[i] = in[i].read();
        #ifdef DEBUG_JOIN
        printf("data.pixel[%d] = %6.2f  ", i, float(data.pixel[i]));
        #endif
      }
      #ifdef DEBUG_JOIN
      printf("\n");
      #endif
      out << data;
    }
  }

  #ifdef DEBUG_JOIN
  printf("JOIN: ends\n");
  #endif
}

// ---------------------------------------------------------------------------------------
// split. Splits incomming pixels grouped in pixel_out_t struct into eight output streams
// of size BLOCK_SIZE elements each.
//
// Arguments:
//   H, W                : sata channel height and width
//   in                  : input stream
//   out0, ... out7      : output streams
//
// The input stream has CPO pixels per data item whereas each output stream has
// BLOCK_SIZE pixels per data item. Therefore, this module reads during BLOCK_SIZE
// cycles the input stream and assigns each pixel from each read data item into
// every output data item to be sent. After those cycles the out data items are
// sent through the corresponding output stream
//
static void split(int H, int W, int *addr_channel, int num_blocks_channel, hls::stream<pixel_out_t> &in, hls::stream<write_block_t> out[CPO]) {

  #ifdef DEBUG_SPLIT
  printf("DEBUG_SPLIT: starts\n");
  printf("DEBUG_SPLIT: num_blocks_channel %d\n", num_blocks_channel);

  #endif

  int num_pixels = H * W;                                       // pixels to receive per channel
  write_block_t cb_[CPO];										// current block buffer
  write_block_t lb_[CPO];										// last block buffer
  int offset_[CPO];												// block offset for incoming pixel
  int current_block_[CPO];										// current block id being built
  int fpa_[CPO];												// first pixel aligned flag
  pixel_out_t data;												// received pixels
  DO_PRAGMA(HLS ARRAY_PARTITION variable=cb_ complete dim=0)
  DO_PRAGMA(HLS ARRAY_PARTITION variable=lb_ complete dim=0)
  DO_PRAGMA(HLS ARRAY_PARTITION variable=offset_ complete dim=0)
  DO_PRAGMA(HLS ARRAY_PARTITION variable=curr_block_ complete dim=0)
  DO_PRAGMA(HLS ARRAY_PARTITION variable=fpa_ complete dim=0)
  DO_PRAGMA(HLS ARRAY_PARTITION variable=data complete dim=0)

  // structs initialization
  for (int cpo=0; cpo<CPO; cpo++) {
	  DO_PRAGMA(HLS UNROLL)
	  offset_[cpo] = addr_channel[cpo] % WRITE_BLOCK_SIZE;
	  fpa_[cpo] = (offset_[cpo] == 0);
	  current_block_[cpo] = 0;
      #ifdef DEBUG_SPLIT
	  printf("DEBUG_SPLIT: cpo %d -> offset %d fpa %d current_block %d\n", cpo, offset_[cpo], fpa_[cpo], current_block_[cpo]);
      #endif
  }

  split_loop:
  for (int r=0; r<num_pixels; r++) {
    DO_PRAGMA(HLS loop_tripcount  min=1 max=WHMAX)
    #pragma HLS PIPELINE II=1

    data = in.read();

    for(int cpo=0; cpo<CPO; cpo++){
      DO_PRAGMA(HLS loop_tripcount  min=1 max=CPO)
      #pragma HLS UNROLL

	  int cpo_prev = (cpo + CPO - 1) % CPO;
      int last_block = current_block_[cpo] == num_blocks_channel-1;

	  if (last_block) {
		  lb_[cpo].pixel[offset_[cpo]] = data.pixel[cpo];
          #ifdef DEBUG_SPLIT
		  printf("writing on lb: cpo %d offset %d pixel %f\n", cpo, offset_[cpo], data.pixel[cpo]);
          #endif
	  } else {
		  if (fpa_[cpo]) {
			  cb_[cpo].pixel[offset_[cpo]] = data.pixel[cpo];
			  #ifdef DEBUG_SPLIT
			  printf("writing on cb: cpo %d offset %d pixel %f\n", cpo, offset_[cpo], data.pixel[cpo]);
              #endif
		  } else {
			  lb_[cpo_prev].pixel[offset_[cpo]] = data.pixel[cpo];
              #ifdef DEBUG_SPLIT
              printf("writing on lb: cpo %d offset %d pixel %f\n", cpo_prev, offset_[cpo], data.pixel[cpo]);
              #endif
		  }
	  }
	  offset_[cpo] = (offset_[cpo] + 1) % WRITE_BLOCK_SIZE;
	  if (offset_[cpo] == 0) {

		  if (fpa_[cpo] && (!last_block)) {
			  out[cpo] << cb_[cpo];
			  current_block_[cpo] = current_block_[cpo] + 1;
              #ifdef DEBUG_SPLIT
              printf("sending block: cpo %d -> ", cpo);
              for (int pp=0; pp<WRITE_BLOCK_SIZE; pp++) printf("%6.4f ", cb_[cpo].pixel[pp]);
              printf("\n");
              #endif
		  } else {
			  fpa_[cpo] = 1;
		  }
	  }
    }
  }

  for (int cpo=0; cpo<CPO; cpo++) {
	DO_PRAGMA(HLS UNROLL)
    out[cpo] << lb_[cpo];
    #ifdef DEBUG_SPLIT
    printf("sending block: cpo %d -> ", cpo);
    for (int pp=0; pp<WRITE_BLOCK_SIZE; pp++) printf("%6.4f ", lb_[cpo].pixel[pp]);
    printf("\n");
    #endif
  }
}

// ---------------------------------------------------------------------------------------
// write_data_channels. Writes data channels from the elements read from an input stream
//
// Arguments:
//   num_pixels          : Number of pixels for each channel
//   ptr                 : pointer to output buffer
//   offset              : offset within output buffer
//   in                  : input streams, one per CPO channel
//   enable              : if not set the module just consumes the input stream and does not write memory, one per CPO channel
//
// On every cycle the module receives BLOCK_SIZE pixels to write into memory
//

static void write_data_channels(int num_pixels, ap_uint<512> *ptr, int *offset_i, hls::stream<write_block_t> in[CPO], int *enable_write) {

  int num_blocks = (num_pixels + WRITE_BLOCK_SIZE - 1) / WRITE_BLOCK_SIZE;

  #ifdef DEBUG_WRITE_DATA
  printf("WRITE_DATA: starts\n");
  printf("WRITE_DATA: num_pixels %d, num_blocks %d\n", num_pixels, num_blocks);
  #endif

write_block_t bx[CPO];
  int block_offset[CPO];
  DO_PRAGMA(HLS ARRAY_PARTITION variable=bx complete)
  DO_PRAGMA(HLS ARRAY_PARTITION variable=block_offset complete)

  write_data_channels_loop_init:
  for (int cpo=0; cpo<CPO; cpo++) {
	DO_PRAGMA(HLS UNROLL)
	block_offset[cpo] = offset_i[cpo] / WRITE_BLOCK_SIZE;
    #ifdef DEBUG_WRITE_DATA
	printf("WRITE_DATA: cpo %d -> offset %d\n", cpo, block_offset[cpo]);
    #endif
  }

  write_data_channels_loop_blocks:
  for (int p = 0; p < num_blocks; p++) {
	DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=WHMAX/WRITE_BLOCK_SIZE)
	write_data_channels_loop_cpo:
	for (int cpo=0; cpo<CPO; cpo++) {
      #pragma HLS pipeline II=1
      bx[cpo] = in[cpo].read();
      if (enable_write[cpo]) {
    	ap_uint<512> data_out;
    	for (int x = 0; x < WRITE_BLOCK_SIZE; x++) {
    		DO_PRAGMA(HLS UNROLL)
    		int first = x * DATA_TYPE_WIDTH;
    		int last = first + DATA_TYPE_WIDTH - 1;
    		data_type datum = bx[cpo].pixel[x];
    		data_out.range(last, first) = *(ap_uint<DATA_TYPE_WIDTH>*)(&datum);
    	}
        ptr[block_offset[cpo]+p] = data_out;
        #ifdef DEBUG_WRITE_DATA
        printf("WRITE_DATA: writting block cpo %d\n", cpo);
        #endif
      }
    }
  }
}

// -------------------------------------------------------------------------------
// relu: module of ReLu function
//
// Arguments:
//   enable_relu: : Flag to enable ReLu function
//   H            : Height of the input channel
//   W            : Width of the input channel
//   in           : input data stream
//   out          : output data stream
//
// This module builds ReLu function by instantiatig streams and
// building the dataflow model with the corresponding modules
//
static void relu(int enable_relu, int H, int W, hls::stream<pixel_out_t> &in, hls::stream<pixel_out_t> &out) {

#ifdef DEBUG_RELU
printf("relu: start\n");
#endif

pixel_out_t data;
int data_size = W * H;
for (int i=0; i < data_size; i++) {
  DO_PRAGMA(HLS loop_tripcount  min=1 max=WHMAX)
  #pragma HLS PIPELINE II=1
  data  = in.read();
  for(int cpo = 0; cpo<CPO; cpo++){
    DO_PRAGMA(HLS loop_tripcount  min=1 max=CPO)
    #pragma HLS UNROLL
    if(enable_relu & (data.pixel[cpo] < 0)) data.pixel[cpo] = data_type(0.f);
  }
  out << data;
}

#ifdef DEBUG_RELU
printf("relu: end\n");
#endif
}

void set_write_enables(int enable_write[CPO], int o_channel, int O) {
  set_write_enables_loop:
  for (int o = 0; o <CPO; o++) {
    DO_PRAGMA(HLS loop_tripcount min=1 max=CPO)
    #pragma HLS UNROLL
    enable_write[o] = (o_channel + o) < O;
  }
}

void set_read_enables(int enable_read[CPI], int I) {
   set_read_enables_loop:
   for (int i = 0; i <CPI; i++) {
     DO_PRAGMA(HLS loop_tripcount min=1 max=CPI)
     #pragma HLS UNROLL
     enable_read[i] = (I >= i+1);
   }
}

void set_reading_channel_offsets(int offset_read_data_channel_i[CPI], int offset_read_data_channel, int channel_offset) {
 set_reading_channel_offsets_loop:
 for(int i=0; i<CPI; i++){
   DO_PRAGMA(HLS loop_tripcount  min=1 max=CPI)
   #pragma HLS UNROLL
   offset_read_data_channel_i[i] = (offset_read_data_channel + i * channel_offset) % READ_BLOCK_SIZE;
 }
}

void set_writing_channel_offsets(int offset_write_data_channel_i[CPO], int global_offset, int channel_offset, int o_channel) {
  set_writing_channel_offsets_loop:
  for(int i=0; i<CPO; i++){
    DO_PRAGMA(HLS loop_tripcount  min=1 max=CPO)
    #pragma HLS UNROLL
    offset_write_data_channel_i[i] = global_offset + (channel_offset * i) + (o_channel * channel_offset);
  }
}

void set_channel_write_blocks(int num_channel_write_blocks[CPO], int addr[CPO], int H, int W) {
  set_channel_write_blocks_loop:
  for(int i=0; i<CPO; i++) {
    #pragma HLS UNROLL
	num_channel_write_blocks[i] = ((H * W) + (addr[i] % WRITE_BLOCK_SIZE) + WRITE_BLOCK_SIZE - 1) / WRITE_BLOCK_SIZE;
  }
}



// ---------------------------------------------------------------------------------------
// padding. Adds padding to the input and forwards it through the output
//
// Arguments:
//   H                 : Height of input channel
//   W                 : Width of input channel
//   I_ITER            : Number of input iterations (I / CPI)
//   in                : input stream
//   out               : output stream
//
static void padding(int H, int W, int I_ITER, int enable_upper_padding, int enable_lower_padding, hls::stream<pixel_in_t> &in, hls::stream<pixel_in_t> &out) {

  #ifdef DEBUG_PADDING
  printf("PADDING: start\n");
  #endif

  int num_iters;
  int h;
  int w;
  pixel_in_t data;
  pixel_in_t zero;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=data complete)
  DO_PRAGMA(HLS ARRAY_PARTITION variable=zero complete)

  padding_cpi_loop:
  for (int cpi=0; cpi<CPI; cpi++){
    DO_PRAGMA(HLS loop_tripcount  min=1 max=CPI)
    #pragma HLS UNROLL
    zero.pixel[cpi] = 0.f;
  }

  num_iters = I_ITER * (H + 2) * (W + 2);
  h = 0;
  w = 0;
  padding_loop:
  for (int i = 0; i < num_iters; i++) {
	DO_PRAGMA(HLS loop_tripcount min=1 max=(I_REFERENCE/CPI) * HMAX * WMAX)
    #pragma HLS pipeline II=1
    int enable1 = enable_upper_padding & (h==0);
	int enable2 = enable_lower_padding & (h == H+1);
	int enable3 = (w == 0);
	int enable4 = (w == W+1);
	if (enable1 | enable2 | enable3 | enable4) data = zero; else data = in.read();
    out << data;
	w = w+1;
	if (w == W+2) {
	  w = 0;
	  h = h + 1;
	  if (h == H+2) {
		h = 0;
	  }
	}
  }

  #ifdef DEBUG_PADDING
  printf("PADDING: end\n");
  #endif
}

// ---------------------------------------------------------------------------------------------------
// cvt: reads an input stream with an image of format (H, W, CPI) and writes an output stream
// in a 2D format based on (KW, KH). (SW=1, SH=1) stride is assumed and (PW=1, PH=1) padding is assumed.
// The function outputs data in the format (KH, KW, CPI).
//
// Arguments:
//   H      : Height of input channel
//   W      : Width of input channel
//   I_ITER : Number of input iterations (I / CPI)
//   in     : input stream (format pixel_in_t)
//   out    : output stream (format frame_t)
//
static void cvt(int H, int W, int I_ITER, hls::stream<pixel_in_t> &in, hls::stream<frame_t> &out) {

  #ifdef DEBUG_CVT
  printf("cvt: start\n");
  #endif

  cvt_i_iter_loop:
  for(int i_iter = 0; i_iter < I_ITER; i_iter++){
    DO_PRAGMA(HLS loop_tripcount  min=1 max=I_REFERENCE/CPI)

    // Now we process the input data and convert the data into frames
    // buffers (keep three rows)
    pixel_in_t buffer0[WMAX+2];
    pixel_in_t buffer1[WMAX+2];
    pixel_in_t buffer2[WMAX+2];
    DO_PRAGMA(HLS ARRAY_PARTITION variable=buffer0 cyclic dim=1 factor=CPI)
    DO_PRAGMA(HLS ARRAY_PARTITION variable=buffer1 cyclic dim=1 factor=CPI)
    DO_PRAGMA(HLS ARRAY_PARTITION variable=buffer2 cyclic dim=1 factor=CPI)

    // frame
    frame_t frame;
    DO_PRAGMA(HLS ARRAY_PARTITION variable=frame)

    // We loop for every incoming pixel
    cvt_loop_1:
    for (int pin_row=0; pin_row < H+2; pin_row++) {
      DO_PRAGMA(HLS loop_tripcount  min=1 max=HMAX+2)
      cvt_loop_2:
      for (int pin_col=0; pin_col < W+2; pin_col++) {
        DO_PRAGMA(HLS loop_tripcount  min=1 max=WMAX+2)
        // get the pixel
        pixel_in_t pixel;
        pixel = in.read();
        // row buffer write (in which buffer row we write the pixel)
        int row0_buffer_write = (pin_row % 3) == 0;
        int row1_buffer_write = (pin_row % 3) == 1;
        // first row buffer
        int row0 = (pin_row <= 2) | ((pin_row % 3) == 2);
        int row1 = !row0 & ((pin_row % 3) == 0);
        // we write the pixel into the buffer
        if (row0_buffer_write) buffer0[pin_col] = pixel; else if (row1_buffer_write) buffer1[pin_col] = pixel; else buffer2[pin_col] = pixel;
        // build the frame
        pixel_in_t p0, p1, p2, p3, p4, p5, p6, p7, p8;
        int shift_frame = (pin_row>1) & (pin_col > 2);
        int send_frame = (pin_row>1) & (pin_col > 1);
        pixel_in_t pixel_b0, pixel_b1, pixel_b2;
        pixel_b0 = buffer0[pin_col];
        pixel_b1 = buffer1[pin_col];
        pixel_b2 = buffer2[pin_col];
        // p0, p1, p2
        if (shift_frame) {p0 = p1;} else if (pin_col==0) {if (row0) p0 = pixel_b0; else if (row1) p0 = pixel_b1; else p0 = pixel_b2;}
        if (shift_frame) {p1 = p2;} else if (pin_col==1) {if (row0) p1 = pixel_b0; else if (row1) p1 = pixel_b1; else p1 = pixel_b2;}
        if (row0) p2 = pixel_b0; else if (row1) p2 = pixel_b1; else p2 = pixel_b2;
        // p3, p4, p5
        if (shift_frame) {p3 = p4;} else if (pin_col==0) {if (row0) p3 = pixel_b1; else if (row1) p3 = pixel_b2; else p3 = pixel_b0;}
        if (shift_frame) {p4 = p5;} else if (pin_col==1) {if (row0) p4 = pixel_b1; else if (row1) p4 = pixel_b2; else p4 = pixel_b0;}
        if (row0) p5 = pixel_b1; else if (row1) p5 = pixel_b2; else p5 = pixel_b0;
        // p6, p7, p8
        if (shift_frame) {p6 = p7;} else if (pin_col==0) {if (row0) p6 = pixel_b2; else if (row1) p6 = pixel_b0; else p6 = pixel_b1;}
        if (shift_frame) {p7 = p8;} else if (pin_col==1) {if (row0) p7 = pixel_b2; else if (row1) p7 = pixel_b0; else p7 = pixel_b1;}
        if (row0) p8 = pixel_b2; else if (row1) p8 = pixel_b0; else p8 = pixel_b1;

        if (send_frame) {
          frame.pixel[0] = p0; frame.pixel[1] = p1; frame.pixel[2] = p2;
          frame.pixel[3] = p3; frame.pixel[4] = p4; frame.pixel[5] = p5;
          frame.pixel[6] = p6; frame.pixel[7] = p7; frame.pixel[8] = p8;
          out << frame;
          #ifdef DEBUG_CVT
          printf("cvt: frame sent:\n");
          for (int cpi=0; cpi<CPI; cpi++) {
            printf("  cpi %d:\n", cpi);
            printf("    %6.4f %6.4f %6.4f\n", float(frame.pixel[0].pixel[cpi]), float(frame.pixel[1].pixel[cpi]), float(frame.pixel[2].pixel[cpi]));
            printf("    %6.4f %6.4f %6.4f\n", float(frame.pixel[3].pixel[cpi]), float(frame.pixel[4].pixel[cpi]), float(frame.pixel[5].pixel[cpi]));
            printf("    %6.4f %6.4f %6.4f\n", float(frame.pixel[6].pixel[cpi]), float(frame.pixel[7].pixel[cpi]), float(frame.pixel[8].pixel[cpi]));
          }
          #endif
        }
      }
    }
  } //i_iter

  #ifdef DEBUG_CVT
  printf("cvt: end\n");
  #endif
}

// ----------------------------------------------------------------------------------------
// mul: This function performs the multiplication of an input frame with the stored kernels
// and sends the produced pixels. Before normal operation it receives its kernels
// Arguments:
//   H     : Height of the input channel
//   W     : Width of the input channel
//   I_ITER: Number of input iterations (I / CPI)
//   in    : input stream with incoming data frames
//   k_in  : input stream with kernels
//   out   : output stream
//
static void mul(int H, int W, int I_ITER, hls::stream<frame_t> &in, hls::stream<kernel_t> &k_in, hls::stream<pixel_out_t> &out) {

  #ifdef DEBUG_MUL
  printf("mul: start\n");
  #endif

  kernel_t kernel;
  kernel_in_t k;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=kernel dim=0 complete)
  #pragma HLS array_partition variable=k dim=0 complete

  frame_t data_in;

  data_type sum[CPO];
  DO_PRAGMA(HLS ARRAY_PARTITION variable=sum dim=0 block factor=CPO)

  pixel_out_t p_out;

  int load_kernel = 0;
  int num_iter = I_ITER * H * W;
  int iter_load_kernel = 0;

  mul_loop_1:
  for(int i = 0; i < num_iter; i++){
    DO_PRAGMA(HLS loop_tripcount  min=1 max=WMAX*HMAX*I_REFERENCE/CPI)
    #pragma HLS PIPELINE II=1
    load_kernel = (iter_load_kernel == 0);
    if (load_kernel){
      kernel = k_in.read();
      #ifdef DEBUG_MUL
      printf("MUL: kernel read\n");
      for(int i=0; i<CPI; i++){
        for(int o=0; o<CPO; o++){
          printf("kernel cpi=%d cpo=%d\n", i, o);
          for (int p=0; p<9; p++){
            printf(" %f ", float(kernel.pixel[o][i][p]));
            if((p+1)%3==0)printf("\n");
          }
          printf("\n");
        }
      }
      #endif
    }

    mul_loop_2:
    for(int i=0; i<CPO; i++){
      DO_PRAGMA(HLS loop_tripcount  min=1 max=CPO)
      #pragma HLS UNROLL
      sum[i] = 0;
    }

    data_in = in.read();

    loop_mul_cpi:
    for (int cpi=0; cpi<CPI; cpi++) {
      DO_PRAGMA(HLS loop_tripcount  min=1 max=CPI)
      #pragma HLS UNROLL
      loop_mul_j:
      for (int j=0; j<KW*KH; j++) {
        DO_PRAGMA(HLS loop_tripcount  min=1 max=KW*KH)
        #pragma HLS UNROLL
        loop_mul_cpo:
        for (int cpo=0; cpo<CPO; cpo++) {
          DO_PRAGMA(HLS loop_tripcount  min=1 max=CPO)
          #pragma HLS UNROLL
          sum[cpo] += data_in.pixel[j].pixel[cpi] * kernel.pixel[cpo][cpi][j];
        }
      }
    }

    for(int i=0; i<CPO; i++){
      DO_PRAGMA(HLS loop_tripcount  min=1 max=CPO)
      #pragma HLS UNROLL
      p_out.pixel[i] = sum[i];
    }
    #ifdef DEBUG_MUL
    for(int i = 0;i<CPO;i++) {
      printf("mult: p_out.pixel[%d] = %6.2f  ", i, float(p_out.pixel[i]));
    }
    printf("\n");
    #endif
    out << p_out;
    iter_load_kernel++;
    if (iter_load_kernel == W*H) iter_load_kernel = 0;
  }

  #ifdef DEBUG_MUL
  printf("mul: end\n");
  #endif
}

// -------------------------------------------------------------------------------
// add: This function performs the addition of all subpixels for the same channel.
// It adds also the corresponding bias.
//
// Arguments:
//   H     : Height of input channel
//   W     : Width of input channel
//   I_ITER: Number of input iterations (I / CPI)
//   in    : input streams data
//   b_in  : input stream bias
//   out   : output stream
//
static void add(int H, int W, int I_ITER, hls::stream<pixel_out_t> &in, hls::stream<pixel_out_t> &b_in, hls::stream<pixel_out_t> &out) {

  #ifdef DEBUG_ADD
  printf("add: start\n");
  #endif

  pixel_out_t bias;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=bias dim=0 complete)

  // number of iterations by CPI || CPO channels
  int num_iterations = W * H;

  // Buffer for all data and CPO channels
  data_type buff_o_channels[CPO][WHMAX];
  DO_PRAGMA(HLS ARRAY_PARTITION variable=buff_o_channels dim=1 complete)


  // We receive bias in packs of CPO
  bias = b_in.read();

  #ifdef DEBUG_ADD
  for (int b=0; b<CPO; b++) {
    printf("Bias[%d] = %6.4f \n", b, float(bias[b]));
  }
  printf("add: bias received\n");
  for(int cpo = 0; cpo<CPO; cpo++){
    printf("Channel cpo = %d: ", cpo);
    for(int it = 0; it<num_iterations; it++){
      printf("%6.2f ", float(buff_o_channels[cpo][it]));
    }
    printf("\n");
  }
  #endif

  // All input data have effect into output add
  add_i_iter_loop:
  for (int i_iter = 0; i_iter < I_ITER; i_iter++){
    DO_PRAGMA(HLS loop_tripcount  min=1 max=I_REFERENCE/CPI)
    pixel_out_t data_out;
    #pragma HLS loop_flatten off
    add_load_data_it_loop:
    for(int it = 0; it<num_iterations; it++){
      DO_PRAGMA(HLS loop_tripcount  min=1 max=WHMAX)
      pixel_out_t data_in;
      data_in = in.read();
      pixel_out_t data;
      add_load_data_cpo_loop:
      for (int cpo=0; cpo<CPO; cpo++) {
        DO_PRAGMA(HLS loop_tripcount  min=1 max=CPO)
        #pragma HLS unroll
        if(i_iter == 0){
          data.pixel[cpo] = bias.pixel[cpo];
        } else {
          data.pixel[cpo] = buff_o_channels[cpo][it];
        }
        buff_o_channels[cpo][it] = data.pixel[cpo] + data_in.pixel[cpo];

        if(i_iter ==(I_ITER-1)){
          data_out.pixel[cpo] = buff_o_channels[cpo][it];
        }
      }
      if(i_iter ==(I_ITER-1)){
        out << data_out;
      }
    }
  } //i_iter

  #ifdef DEBUG_ADD
  for (int cpo=0; cpo<CPO; cpo++) {
    printf("CH %d: ", cpo);
    for (int it=0; it<num_iterations; it++) {
      printf("%6.2f ", float(buff_o_channels[cpo][it]));
    }
    printf("\n");
  }
  #endif

  #ifdef DEBUG_ADD
  printf("add: end\n");
  #endif
}


// -------------------------------------------------------------------------------
// conv: Convolutional kernel
//
// Arguments:
//   H      : Height of the input channel
//   W      : Width of the input channel
//   I_ITER : Number of input iterations (I / CPI)
//   in     : input data stream
//   k_in   : input kernel stream
//   b_in   : input bias stream
//   out    : output data stream
//
// This module builds the convolutional operation by instantiating streams and
// building the dataflow model with the corresponding modules
//
static void conv(int H, int W, int I_ITER, int enable_upper_padding, int enable_lower_padding, hls::stream<pixel_in_t> &in, hls::stream<kernel_t> &k_in, hls::stream<pixel_out_t> &b_in, hls::stream<pixel_out_t> &out) {

  // streams
  static hls::stream<pixel_in_t>  str_pad_cvt;  // padding->cvt
  static hls::stream<frame_t>     str_cvt_mul;  // cvt->mul
  static hls::stream<pixel_out_t> str_mul_add;  // mul->add
  DO_PRAGMA(HLS stream variable=str_pad_cvt depth=CPO)
  DO_PRAGMA(HLS stream variable=str_cvt_mul depth=CPO)
  DO_PRAGMA(HLS stream variable=str_mul_add depth=CPO)


  // topology
  #pragma HLS dataflow
  padding(H, W, I_ITER, enable_upper_padding, enable_lower_padding, in, str_pad_cvt);            // padding
  cvt(H, W, I_ITER, str_pad_cvt, str_cvt_mul);       // cvt
  mul(H, W, I_ITER, str_cvt_mul, k_in, str_mul_add); // mul
  add(H, W, I_ITER, str_mul_add, b_in, out);         // add
}

extern "C" {

void k_conv2D(ap_uint<512> *ptr_data, int H, int W, int rows, int I, int O, int I_ITER, int O_ITER, int enable_relu,
              data_type *ptr_kernel, pixel_out_t *ptr_bias, ap_uint<512> *ptr_out, int global_offset, int enable_upper_padding, int enable_lower_padding) {
	#pragma HLS INTERFACE m_axi port=ptr_data   offset=slave bundle=gmem
	#pragma HLS INTERFACE m_axi port=ptr_kernel depth=10 offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=ptr_bias   offset=slave bundle=gmem2
	#pragma HLS INTERFACE m_axi port=ptr_out    offset=slave bundle=gmem3

  #ifdef DEBUG_VERBOSE
  printf("kernel starts...\n");
  #endif

  o_iter_loop:
  for (int o_iter = 0; o_iter<O_ITER; o_iter++) {
	DO_PRAGMA(HLS loop_tripcount min=1 max=MAX_OITER)
	#pragma HLS dataflow

	int o_channel = o_iter << LOG2_CPO;  // current output channel (first one in this iteration)



    // input and output streams
    static hls::stream<pixel_in_t>   out_read_data;
    static hls::stream<pixel_in_t>   out_read_data_2;
    static hls::stream<kernel_t>     out_read_kernel;
    static hls::stream<pixel_out_t>  out_read_bias;
    static hls::stream<pixel_out_t>  out_conv;
    static hls::stream<pixel_out_t>  out_relu;
    static hls::stream<read_block_t> stream_data_ch_0[CPI];
    static hls::stream<data_type>    stream_data_ch_1[CPI];
    static hls::stream<write_block_t> out_write_channel[CPO];

    // variables
    int enable_read[CPI];
    int enable_write[CPO];
    int offset_read_data_channel_i[CPI];
    int offset_write_data_channel_i[CPO];
    int num_channel_write_blocks[CPO];
    int corrected_offset         = (enable_upper_padding==0)? W : 0;
    int channel_offset           = (W * H);
    int num_extra_rows           = (enable_lower_padding == 0) + (enable_upper_padding == 0);
    int offset_read_data_channel = global_offset - corrected_offset;
    int channel_size             = H * W;
    int read_pixels              = W * (rows + num_extra_rows);
    int write_pixels             = rows * W;
    int channel_blocks           = (read_pixels + READ_BLOCK_SIZE - 1) / READ_BLOCK_SIZE;
    int res_blocks               = channel_size % READ_BLOCK_SIZE;
    int offset_bias              = o_iter;
    int offset_kernel            = o_iter * (I < CPI ? CPI : I) * CPO * 9;
    #pragma HLS array_partition variable=enable_read dim=0 complete
    #pragma HLS array_partition variable=enable_write dim=0 complete
    DO_PRAGMA(HLS ARRAY_PARTITION variable=offset_read_data_channel_i dim=0 complete)
    DO_PRAGMA(HLS ARRAY_PARTITION variable=offset_write_data_channel_i dim=0 complete)
	DO_PRAGMA(HLS ARRAY_PARTITION variable=num_channel_write_blocks dim=0 complete)

    // we compute the enable_read signals
    set_read_enables(enable_read, I);

    // we compute the enable_write signals
    set_write_enables(enable_write, o_channel, O);

    // channel offsets for reading
    set_reading_channel_offsets(offset_read_data_channel_i, offset_read_data_channel, channel_offset);

    // channel offsets for writing
    set_writing_channel_offsets(offset_write_data_channel_i, global_offset, channel_offset, o_channel);

    // channel write blocks
    set_channel_write_blocks(num_channel_write_blocks, offset_write_data_channel_i, H, W);

    read_bias(offset_bias, ptr_bias, out_read_bias);
    read_kernel(I_ITER, offset_kernel, ptr_kernel, out_read_kernel);
    read_data_channels(H, W, rows, I_ITER, ptr_data, offset_read_data_channel, num_extra_rows, channel_blocks, stream_data_ch_0, enable_read);
    ch_serialize_and_filter<CPI>(I_ITER, read_pixels, channel_blocks, channel_size, offset_read_data_channel_i, stream_data_ch_0, stream_data_ch_1, enable_read);
    join(rows, W, I_ITER, num_extra_rows, stream_data_ch_1,  out_read_data);
    conv(rows, W, I_ITER, enable_upper_padding, enable_lower_padding, out_read_data, out_read_kernel, out_read_bias, out_conv);
    relu(enable_relu, rows, W, out_conv, out_relu);
    split(rows, W, offset_write_data_channel_i, channel_blocks, out_relu, out_write_channel);

    //split(rows, W, out_relu, out_write_channel);
    write_data_channels(write_pixels, ptr_out, offset_write_data_channel_i, out_write_channel, enable_write);

 }

 #ifdef DEBUG_VERBOSE
 printf("kernel finishes\n");
 #endif

}

} // extern "C"
