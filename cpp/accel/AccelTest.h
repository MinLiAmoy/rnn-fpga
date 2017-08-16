#ifndef ACCEL_ACCEL_TEST_H
#define ACCEL_ACCEL_TEST_H

#include "Typedefs.h"
#include "Accel.h"
#include "AccelPrint.h"
#include <cstdlib>

const unsigned N_LAYERS = 3;
const unsigned N_W_LAYERS = 8;
/*const unsigned L_CONV = 6;
const unsigned S_tab[] =  { 32,  32,  16,  16,   8,   8,    4,    1,    1,   1};    // ML: dim of output
const unsigned M_tab[] =  {  3, 128, 128, 256, 256, 512, 8192, 1024, 1024};
const unsigned N_tab[] =  {128, 128, 256, 256, 512, 512, 1024, 1024,   10};     // ML: the dim of inputs and outputs
const unsigned T_tab[] =  {  0,   1,   1,   1,   1,   1,    2,    2,    3};     // ML: the idx of LayerTypeEnum to the sequential layers
*/
//const unsigned S_tab[] =  {  1,   1,   1,  1};
const unsigned M_tab[] =  { 64, 128, 128};
const unsigned N_tab[] =  {128, 128,  64};
const unsigned T_tab[] =  {  0,   1,   2};
const unsigned widx_tab[] = {  0, 1, 2, 3, 4, 5, 6, 7, 18, 19, 20, 21, 22, 23, 24, 25, 36}//ML: need to be modifed. the dim should be N_W_LAYERS*N_LAYERS

// ML: n_file in param arc is 29
/*const unsigned widx_tab[] = {0,   3,   6,   9,  12,  15,   18,   21,   24};     // ML: the idx of weights array in arc
const unsigned kidx_tab[] = {1,   4,   7,  10,  13,  16,   19,   22,   25};     // ML: the idx of k array in arc
const unsigned hidx_tab[] = {2,   5,   8,  11,  14,  17,   20,   23,   26};     // ML: the idx of h array in arc
const unsigned pool_tab[] = {0,   1,   0,   1,   0,   1,    0,    0,    0};*/
const char vocab[] = {'\n', '!', ' ', '$', "'", '&', '-', ',', '.', ';', ':', '?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
                       'I', 'H', 'K', 'J', 'M', 'L', 'O', 'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z', 'a',
                       'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'p', 'q', 's', 'r', 'u', 't'
                       'w', 'v', 'y', 'x', 'z'}

// layer_idx goes from 1 to 9
bool layer_is_rnn(unsigned layer_idx);
/*bool layer_is_conv(unsigned layer_idx);
bool layer_is_binconv(unsigned layer_idx);
bool layer_is_fpconv(unsigned layer_idx);*/
bool layer_is_last(unsigned layer_idx);
bool layer_wt_size(unsigned layer_idx);
bool layer_kh_size(unsigned layer_idx);

// number of Words allocated to store n weights
unsigned WTS_TO_WORDS(const unsigned n);
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
void set_weight_array(Word* w, const float* wts, unsigned layer_idx);
void set_weight_array_rnn(Word* w, const float* wts, unsigned layer_idx, unsigned weight_idx);
void set_conv_weight_array(Word* w, const float* wts, unsigned size);
void set_dense_weight_array(Word* w, const float* wts, unsigned M, unsigned N);

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
);

#endif
