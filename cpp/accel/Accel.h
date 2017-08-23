#ifndef ACCEL_ACCEL_H
#define ACCEL_ACCEL_H

#include <cstddef>
#include <hls_video.h>
#include <hls_stream.h>
#include <stdlib.h>   // include this before sds_lib.h for size_t

#include "Typedefs.h"
#include "Debug.h"
#include "Common.h"

#ifdef __SDSCC__
  #include "sds_lib.h"
  #define MEM_ALLOC(size) sds_alloc(size)
  #define MEM_FREE(ptr) sds_free(ptr)
#else
  #define MEM_ALLOC(size) malloc(size)
  #define MEM_FREE(ptr) free(ptr)
#endif

//-------------------------------------------------------------------
// Constants
//-------------------------------------------------------------------

// ML: define the param of Recurrent Neural Network
const unsigned HID_SIZE = 128;
const unsigned DATA_SIZE = 16;
const unsigned DATA_PER_WORD = 4;
const unsigned VOCAB_SIZE = 64;
const unsigned WT_SIZE = 2;
//

const unsigned WORD_SIZE = 64;

const unsigned WT_L = (128 + 128)* 3 * 128; // parameter to control wt mem size
const unsigned BIAS_L = 128 * 3; // ML: parameter to control bias memsize


const unsigned WT_WORDS = WT_L * WT_SIZE / WORD_SIZE; // ML: beyond the mem on chip?
const unsigned BIAS_WORDS = BIAS_L * WT_SIZE / WORD_SIZE; 


// ML: mem of input data and output data
const unsigned DMEM_WORDS = 64/DATA_PER_WORD;
const unsigned DMEM_O_WORDS = 64/DATA_PER_WORD; 


//-------------------------------------------------------------------
// Typedefs
//-------------------------------------------------------------------
enum RLayerTyprEnum {LAYER_GRU1, LAYER_GRU2, LAYER_DENSE};
//
typedef ap_int<WORD_SIZE> Word;

typedef ap_uint<16> Address;
/*typedef ap_int<12> ConvSum;
typedef ap_int<5> ConvOut;
typedef ap_uint<10> IdxType;
typedef ap_fixed<16,4> C1Comp;    // ML: -h/k be quantized to be 16 bits fixed-point on the fpconv layer
typedef ap_int<16> NormComp;    // ML: -h/k be quantized to be 16 bits int
typedef ap_int<16> DenseSum;
typedef ap_fixed<16,12> DenseNorm;

typedef ap_fixed<20,2, AP_RND> C1InputType;  // ML: input pixel are 20-bit fixed-point
typedef ap_fixed<24,6, AP_RND> C1ConvType;*/


typedef ap_fixed<16,8> DATA;    // ML: can do the exp operation


//-------------------------------------------------------------------
// Accelerator synthesizable top-level function
//-------------------------------------------------------------------
/*#pragma SDS data copy(dmem_i[0:input_words], dmem_o[0:output_words])
#pragma SDS data access_pattern(dmem_i:SEQUENTIAL, dmem_o:SEQUENTIAL)
#pragma SDS data access_pattern(wt_i:SEQUENTIAL, kh_i:SEQUENTIAL)
#pragma SDS data mem_attribute(dmem_i:PHYSICAL_CONTIGUOUS, dmem_o:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(wt_i:PHYSICAL_CONTIGUOUS, kh_i:PHYSICAL_CONTIGUOUS)
#pragma SDS data data_mover(dmem_i:AXIDMA_SIMPLE, dmem_o:AXIDMA_SIMPLE)
#pragma SDS data data_mover(wt_i:AXIDMA_SIMPLE, kh_i:AXIDMA_SIMPLE)
void top(
    Word wt_i[WT_WORDS],
    Word kh_i[KH_WORDS],
    Word dmem_i[DMEM_WORDS],
    Word dmem_o[DMEM_O_WORDS],
    const Address    n_inputs,
    const Address    n_outputs,
    const Address    input_words,
    const Address    output_words,
    const ap_uint<3> layer_mode,  // [0]='new layer', [2:1]='conv1,conv,dense'
    const ap_uint<1> dmem_mode,   // 0 means dmem[0] is input
    const ap_uint<2> width_mode,  // 0=8'b, 1=16'b, 2=32'b
    const ap_uint<2> norm_mode    // 0='do nothing', 1='do norm', 2='do pool'
);*/

#endif
