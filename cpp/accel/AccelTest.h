#ifndef ACCEL_ACCEL_TEST_H
#define ACCEL_ACCEL_TEST_H

#include <cstddef>
#include "Typedefs.h"
#include "Accel.h"
//#include "AccelPrint.h"
#include <cstdlib>

// ML: num of layer of RNN
const unsigned N_LAYERS = 3;
// ML: num of weight arrays of every gate in rnn layer
const unsigned N_W_LAYERS = 4;


const unsigned M_tab[] =  { 64, 128, 128};  // input num
const unsigned N_tab[] =  {128, 128,  64};  // output num 
const unsigned T_tab[] =  {  0,   1,   2};  // type of 
const unsigned widx_tab[] = {  0, 1, 3, 4, 6, 7, 9, 10, 14, 15, 17, 18, 20, 21, 23, 24, 28}; // idx of each weight array in zip arc
const unsigned bidx_tab[] = {  2, 5, 8, 11, 16, 19, 22, 25, 29};    // idx of each bias array in zip arc

// num of elements in vocab
const char vocab[] = {'\n', ' ', '!', '$', '&', '\'', ',', '-', '.', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                   'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a',
                   'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                   'v', 'w', 'x', 'y', 'z'};

// layer_idx goes from 1 to 9
bool layer_is_rnn(unsigned layer_idx);
bool layer_is_last(unsigned layer_idx);

// Simple log function, only works for powers of 2
unsigned log2(unsigned x);

//------------------------------------------------------------------------
// Set an array of ap_int's using some data, used to binarize test
// inputs and outputs
//------------------------------------------------------------------------
template<typename T1, typename T2>
void set_bit_array(T1 array[], const T2* data, unsigned size) {
  for (unsigned i = 0; i < size; ++i) {
    set_bit(array, i, (data[i]>=0) ? Bit(0) : Bit(-1));
  }
}

//------------------------------------------------------------------------
// Functions used to preprocess params and inputs
//------------------------------------------------------------------------

void set_rnn_weight_array(Word* w, const float* wts_in, const float* wts_hid, unsigned layer_idx, unsigned weight_idx);
void set_rnn_bias_array(Word* b, const float* bias, unsigned layer_idx, unsigned weight_idx);
void set_dense_weight_array(Word* w, const float* wts, unsigned layer_idx);
void set_dense_bias_array(Word* b, const float* bias, unsigned layer_idx);


/*
void set_bnorm_array(Word* kh, const float* k, const float* h, unsigned layer_idx);
void set_bnorm_array1(Word* kh, const float* k, const float* h, unsigned layer_idx, unsigned N);
void set_bnorm_array2(Word* kh, const float* k, const float* h, unsigned N);

void binarize_input_images(Word* dmem_i, const float* inputs, unsigned S);

//------------------------------------------------------------------------
// Padded convolution (used for golden reference)
//------------------------------------------------------------------------
void padded_conv(Word in[], Word w[], Word out[], unsigned M, unsigned S);

//------------------------------------------------------------------------
// Helper test function for the accelerator
// This function calls the accelerator, then runs a check of the results
// against conv_ref (if not NULL) and bin_ref.
//------------------------------------------------------------------------
void test_conv_layer(
    Word* weights,
    Word* kh,
    Word* data_i,
    Word* data_o,
    Word* conv_ref,
    Word* bin_ref,
    const unsigned M,
    const unsigned N,
    const unsigned S,
    const ap_uint<1> conv_mode=1,
    const ap_uint<1> max_pool=0
);

void test_dense_layer(
    Word* weights,
    Word* kh,
    Word* data_i,
    Word* data_o,
    Word* bin_ref,
    const unsigned M,   // pixels
    const unsigned N    // pixels
);*/

#endif
