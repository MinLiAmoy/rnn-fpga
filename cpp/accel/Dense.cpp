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
DATA sigmoid(
  const DATA in
  ) {
  DATA out;
  out = 1/(1+exp(-x));
  return out;
}

DATA tanh(
  const DATA in
  ) {
  DATA out;
  out = (exp(x) - exp(-x))/(exp(x)+exp(-x));
  return out;
}


DATA dotproduct_m(
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
// ML: the size of in is M/DATA_PER_WORD = M/4 words; the size of out is N/DATA_PER_WORD!
void dense_layer_cpu(
    //const Word*  wt,
    //const float* k_data,
    //const float* h_data,
    const Word* data_i,
    Word* data_o,
    unsigned layer_idx,
    const Address inputs_words,
    const Address outputs_words,
    //ap_uint<1> dmem_mode
    AccelSchedule& s 
) {
  //t_dense.start();
  static Word dmem[5][HIDDEN_STATE/DATA_PER_WORD] = {0};

  

  M = s.n_inputs;
  N = s.n_outputs;

  ap_uint<1> d_i_idx = dmem_mode;
  ap_uint<1> d_o_idx = ~dmem_mode;

  Word in[(M+N)/DATA_PER_WORD];
  DATA gate[4][N];  // ML: input, forget, cell(tanh), output

  if (layer_idx < 2) {
    for (unsigned i = 0; i < N+M; i++) {
      if ((i < N) & (layer_idx == 0) & (inputs_words != 0) ) {
        in[i/DATA_PER_WORD] = data_i[i/DATA_PER_WORD];
      }
      else if ((i<N) & (layer_idx == 0) & (inputs_words == 0)) {
        in[i/DATA_PER_WORD] = dmem[0][i/DATA_PER_WORD];
      }
      else if ((i < N) & (layer_idx == 1)) {
        in[i/DATA_PER_WORD] = dmem[1][i/DATA_PER_WORD]; // *ML:is d_i_idx?
      }
      else if ((i >= N) & (layer_idx == 0)) {
        in[i/DATA_PER_WORD] = dmem[1][(i-N)/DATA_PER_WORD];
      }
      else{
        in[i/DATA_PER_WORD] = dmem[3][(i-N)/DATA_PER_WORD];
      }
    }
  }
  
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
      data_o[n/DATA_PER_WORD] = out_wrd;
      dmem[0][n/DATA_PER_WORD] = out_wrd; // ML: dont need another data buffer?
    }
  } else {
    for (unsigned n = 0; n < 4*N; n+=WORD_SIZE) {
      Word out_wrd[WORD_SIZE/DATA_PER_WORD] = {0};
      for (unsigned nb = 0; nb < WORD_SIZE; ++nb) {
        DATA sum = dotproduct_m(in, wt_i, M+N, n+nb);
        out_wrd[nb/WORD_SIZE]((nb%DATA_PER_WORD+1)*16-1, (nb%DATA_PER_WORD)*16-1) = sum(15,0);
        /*float res = static_cast<float>(sum) * k_data[n+nb] + h_data[n+nb];
        if (res < 0) 
          out_wrd[nb] = 1;*/
      }
      gate[n](15,0) = out_wrd(15,0);
    }

    for (unsigned n = 0; n < 4N; n++) {
      unsigned gate_idx = n / N;

      DATA temp;

      if (gate_idx != 2) {
        temp = sigmoid(gate[gate_idx][n%4]);
        gate[gate_idx][n%4] = temp;
      }
      else {
        temp = tanh(gate[gate_idx][n%4]);
        gate[gate_idx][n%4] = temp;
      }
    }

    for (unsigned n = 0; n < N; n++) {
      DATA cell;
      DATA cell_pre;
      DATA hidden;
      cell_pre(15,0) = dmem[layer_idx*2][n/DATA_PER_WORD]((n%DATA_PER_WORD+1)*16-1, (n%DATA_PER_WORD)*16-1)
      // ML: new cell state
      cell = gate[1][n]*cell_pre + gate[0][n]*gate[2][n];
      hidden = gate[3][n]*tanh(cell);
      dmem[layer_idx*2][n/DATA_PER_WORD]((n%DATA_PER_WORD+1)*16-1, (n%DATA_PER_WORD)*16-1) = cell(15,0);
      dmem[layer_idx*2 - 1][n/DATA_PER_WORD]((n%DATA_PER_WORD+1)*16-1, (n%DATA_PER_WORD)*16-1) = hidden(15,0);
    }


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
