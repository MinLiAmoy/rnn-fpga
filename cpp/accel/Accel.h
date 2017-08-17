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

// ML: define
const unsigned Hid_SIZE = 128;
//

const unsigned CONVOLVERS = 2;

const unsigned WORD_SIZE = 64;
const unsigned WT_SIZE = 9;   // ML: a 3*3 filter contains 9 parameters
const unsigned CONV_W_PER_WORD = 7;   // ML: 64/9 = 7, so a word contains 7 sets of weights
const unsigned CONV1_W_PER_WORD = 4;
const unsigned KH_PER_WORD = 4;   // ML: 4 -k/h per word
const unsigned BYTE_SIZE = 8;
const unsigned K = 3;
const unsigned WT_L         = (128 + 128)*4*128; // parameter to control wt mem size
const unsigned C_WT_WORDS   = ((WT_L+CONV_W_PER_WORD-1)/CONV_W_PER_WORD + CONVOLVERS-1) / CONVOLVERS;  // wt words per convolver
//const unsigned WT_WORDS     = C_WT_WORDS*CONVOLVERS;
const unsigned WT_WORDS = (128 + 128) * 4 * 128/ WORD_SIZE; // ML: beyond the mem on chip?

const unsigned KH_WORDS     = WT_L/128*16 / WORD_SIZE;

//const unsigned DMEM_WORDS   = 128*32*32 / WORD_SIZE;    // ML: input data memory words, 128*32*32 is the largest data size(the second layer input)
// ML
const unsigned DMEM_WORDS = 64/DATA_PER_WORD;
const unsigned DMEM_O_WORDS = 64/DATA_PER_WORD; // ML: may need to modified again

const unsigned C_DMEM_WORDS = DMEM_WORDS / CONVOLVERS;
//const unsigned DMEM_O_WORDS = 512*4*4 / WORD_SIZE;    // ML: output data memory words, 512*4*4 is the last conv layer output data size 
const unsigned DB_MEM_WORDS = 32*32;

const unsigned PIX_PER_PHASE = 2*32*32;

const unsigned MAX_WIDTH = WORD_SIZE;
const unsigned BANK_WIDTH = 8;
const unsigned LOG_BANK_WIDTH = 3;

const unsigned CONV_ROWS = 3;
const unsigned CONV_COLS = BANK_WIDTH+2;    // ML: 2->0 padding
const unsigned CONV_BANKS = WORD_SIZE / BANK_WIDTH;

const unsigned DATA_SIZE = 16;
const unsigned DATA_PER_WORD = 4;

//-------------------------------------------------------------------
// Typedefs
//-------------------------------------------------------------------
//enum LayerTypeEnum {LAYER_CONV1, LAYER_CONV, LAYER_DENSE, LAYER_LAST};    // ML: project to 0,1,2,3
// ML: define
enum RLayerTyprEnum {LAYER_RNN1, LAYER_RNN2, LAYER_LAST};
//
typedef ap_int<WORD_SIZE> Word;
typedef ap_int<WT_SIZE> WtType;   // ML: WtType ... ap_int<9>
typedef ap_uint<16> Address;
typedef ap_int<12> ConvSum;
typedef ap_int<5> ConvOut;
typedef ap_uint<10> IdxType;
typedef ap_fixed<16,4> C1Comp;    // ML: -h/k be quantized to be 16 bits fixed-point on the fpconv layer
typedef ap_int<16> NormComp;    // ML: -h/k be quantized to be 16 bits int
typedef ap_int<16> DenseSum;
typedef ap_fixed<16,12> DenseNorm;

typedef ap_fixed<20,2, AP_RND> C1InputType;  // ML: input pixel are 20-bit fixed-point
typedef ap_fixed<24,6, AP_RND> C1ConvType;

// ML:define
typedef ap_fixed<16,8> DATA;    // ML: can do the exp operation

//-------------------------------------------------------------------
// Template functions
//-------------------------------------------------------------------
template<typename T>
void load_kh(T& comp, const Word kh_mem[KH_WORDS], Address idx) {
  Word kh_word = kh_mem[idx/KH_PER_WORD];
  IdxType off = idx % KH_PER_WORD;    // ML: KH_PER_WORD is 4 and every KH is 16 bit fixed point format indeed!
  if (off == 0)
    comp(15,0) = kh_word(15, 0);
  else if (off == 1)
    comp(15,0) = kh_word(31,16);
  else if (off == 2)
    comp(15,0) = kh_word(47,32);
  else
    comp(15,0) = kh_word(63,48);
}

//-------------------------------------------------------------------
// Accelerator synthesizable top-level function
//-------------------------------------------------------------------
#pragma SDS data copy(dmem_i[0:input_words], dmem_o[0:output_words])
#pragma SDS data access_pattern(dmem_i:SEQUENTIAL, dmem_o:SEQUENTIAL)
#pragma SDS data access_pattern(wt_i:SEQUENTIAL, kh_i:SEQUENTIAL)
#pragma SDS data mem_attribute(dmem_i:PHYSICAL_CONTIGUOUS, dmem_o:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(wt_i:PHYSICAL_CONTIGUOUS, kh_i:PHYSICAL_CONTIGUOUS)
#pragma SDS data data_mover(dmem_i:AXIDMA_SIMPLE, dmem_o:AXIDMA_SIMPLE)
#pragma SDS data data_mover(wt_i:AXIDMA_SIMPLE, kh_i:AXIDMA_SIMPLE)
void top(
    Word wt_i[WT_WORDS],
    //Word kh_i[KH_WORDS],
    Word dmem_i[DMEM_WORDS],
    Word dmem_o[DMEM_O_WORDS],
    const Address    n_inputs,
    const Address    n_outputs,
    const Address    input_words,
    const Address    output_words,
    const ap_uint<3> layer_mode,  // [0]='new layer', [2:1]='conv1,conv,dense'
    const ap_uint<1> dmem_mode,   // 0 means dmem[0] is input
    //const ap_uint<2> width_mode,  // 0=8'b, 1=16'b, 2=32'b
    //const ap_uint<2> norm_mode    // 0='do nothing', 1='do norm', 2='do pool'
);

#endif
