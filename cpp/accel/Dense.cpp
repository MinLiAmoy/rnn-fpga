#include "Dense.h"
#include "Timer.h"

const static Word m1("0x5555555555555555", 16);
const static Word m2("0x3333333333333333", 16);
const static Word m4("0x0f0f0f0f0f0f0f0f", 16);
const static Word h01("0x0101010101010101", 16);
static Timer t_dense("dense");
static Timer t_last ("last");

// -----------------------------------------------------------------------
// Performs dense dot product on M input bits, n*M is the weight offset
// -----------------------------------------------------------------------
// ML: the size of in[] is like 1*M and the size of w is n*M where M%word_size = 0
// ML: a call of dotproduct_m can compute a dotproduct using index of n
int dotproduct_m(
    const Word* in,
    const Word* w,
    const unsigned M,
    const unsigned n
) {
  assert (M % WORD_SIZE == 0);
  DATA sum = 0;

  // Loop across in the inputs in batches of WORD_SIZE
  for (unsigned m = 0; m < M; m+=WORD_SIZE) {

    DATA *in_wrd;
    for (unsigned i = 0; i < WORD_SIZE; ++DATA_PER_WORD) {
      in_wrd[i](15,0)   = in[(m + i)/4](15,0);
      in_wrd[i+1](15,0) = in[(m + i)/4](31,16);
      in_wrd[i+2](15,0) = in[(m + i)/4](47,32);
      in_wrd[i+3](15,0) = in[(m + i)/4](63,48);
    }
    const Word wt_wrd = w[(n*M+m)/WORD_SIZE];

    /*Word x = wt_wrd ^ in_wrd;

    // count_set bit for 64 bits, returns 2*cnt
    x -= (x >> 1) & m1;
    x = (x & m2) + ((x >> 2) & m2);
    x = (x + (x >> 4)) & m4;
    x += x >> 8;
    x += x >> 16;
    x += x >> 32;
    x = x & 0x7f;

    sum += WORD_SIZE - (x<<1).to_int();*/
    for (i = 0; i < WORD_SIZE; ++i) {
      if (wt_wrd[i] > 0)
        sum += DATA[i];
      else
        sum -= DATA[i];
    }
  }
  return sum;
}

// -----------------------------------------------------------------------
// Internal dense layer
// -----------------------------------------------------------------------
// ML: k, h is the coefficient of BNN!
// ML: the size of in is M*DATA_PER_WORD = 4M words; the size of out is N*DATA_PER_WORD!
void dense_layer_cpu(
    //const Word*  wt,
    //const float* k_data,
    //const float* h_data,
    const Word* data_i,
    Word* data_o,
    unsigned layer_idx,
    const Address inputs_words,
    const Address outputs_words,
    ap_uint<1> dmem_mode
    AccelSchedule& s 
) {
  //t_dense.start();
  static Word dmem[2][6][HIDDEN_STATE/DATA_PER_WORD] = {0};

  if inp

  M = s.n_inputs;
  N = s.n_inputs;

  ap_uint<1> d_i_idx = dmem_mode;
  ap_uint<1> d_o_idx = ~dmem_mode;
  
  static Word* wt_i = (Words*) MEM_ALLOC( WT_WORDS*sizeof(Word));

  for (unsigned j = 0; j < WT_WORDS; ++j)
      wt_i[j] = s.wt[j];

  if (layer_idx == LAYER_LAST){
    for (unsigned n = 0; n < N; n+=WORD_SIZE) {
      Word out_wrd[WORD_SIZE/DATA_PER_WORD] = {0};
      for (unsigned nb = 0; nb < WORD_SIZE; ++nb) {
        DATA sum = dotproduct_m(dmem[d_i_idx][3], wt_i, M, n+nb);
        out_wrd[nb/WORD_SIZE]((nb%DATA_PER_WORD+1)*16-1, (nb%DATA_PER_WORD)*16-1) = sum(15,0);
        /*float res = static_cast<float>(sum) * k_data[n+nb] + h_data[n+nb];
        if (res < 0)
          out_wrd[nb] = 1;*/
      }
      data_o[n/WORD_SIZE/DATA_PER_WORD] = out_wrd;
    }
  } else {
    for (unsigned n = 0; n < N; n+=WORD_SIZE) {
      Word out_wrd[WORD_SIZE/DATA_PER_WORD] = {0};
      for (unsigned nb = 0; nb < WORD_SIZE; ++nb) {
        DATA sum = dotproduct_m(dmem[d_i_idx][3], wt_i, M, n+nb);
        out_wrd[nb/WORD_SIZE]((nb%DATA_PER_WORD+1)*16-1, (nb%DATA_PER_WORD)*16-1) = sum(15,0);
        /*float res = static_cast<float>(sum) * k_data[n+nb] + h_data[n+nb];
        if (res < 0)
          out_wrd[nb] = 1;*/
      }
      
    }
  }



  for (unsigned n = 0; n < N; n+=WORD_SIZE) {
    Word out_wrd[WORD_SIZE/DATA_PER_WORD] = {0};
    for (unsigned nb = 0; nb < WORD_SIZE; ++nb) {
      DATA sum = dotproduct_m(in, wt, M, n+nb);
      out_wrd[nb/WORD_SIZE]((nb%DATA_PER_WORD+1)*16-1, (nb%DATA_PER_WORD)*16-1) = sum(15,0);
      /*float res = static_cast<float>(sum) * k_data[n+nb] + h_data[n+nb];
      if (res < 0)
        out_wrd[nb] = 1;*/
    }
    out[n/WORD_SIZE/DATA_PER_WORD] = out_wrd;
  }

  //t_dense.stop();
}

// -----------------------------------------------------------------------
// Final dense layer
// -----------------------------------------------------------------------
// ML: N is the dimension of labels and the funciton outputs the predicition!
int last_layer_cpu(
    const Word*  wt,
    const float* k_data,
    const float* h_data,
    const Word* in,
    const unsigned M,
    const unsigned N
) {
  t_last.start();

  int pred = -1;
  float maxval = 0;

  for (unsigned n = 0; n < N; ++n) {
    int sum = dotproduct_m(in, wt, M, n);
    float val = static_cast<float>(sum) * k_data[n] + h_data[n];
    if (pred == -1 || val > maxval) {
      pred = n;
      maxval = val;
    }
  }

  t_last.stop();
  return pred;
}
