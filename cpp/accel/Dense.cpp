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
  out = 1/(1+hls::exp((ap_fixed<16,8>) -in));
  return out;
}

DATA tanh(
  const DATA in
  ) {
  DATA out;
  out = (hls::exp((ap_fixed<16,8>) in) - hls::exp((ap_fixed<16,8>) -in)) / (hls::exp((ap_fixed<16,8>) in) + hls::exp((ap_fixed<16,8>) -in));
  return out;
}


DATA dotproduct_m(
    const Word in[2*HID_SIZE],
    const Word w[WT_WORDS],
    const unsigned M,
    const unsigned n
) {
  assert (M % WORD_SIZE == 0);
  DATA sum = 0;

  // Loop across in the inputs in batches of WORD_SIZE
  for (unsigned m = 0; m < M; m+=WORD_SIZE) {

    DATA in_wrd[WORD_SIZE];
    for (unsigned i = 0; i < WORD_SIZE; i+=DATA_PER_WORD) {
      in_wrd[i](15,0)   = in[(m + i)/4](15,0);
      in_wrd[i+1](15,0) = in[(m + i)/4](31,16);
      in_wrd[i+2](15,0) = in[(m + i)/4](47,32);
      in_wrd[i+3](15,0) = in[(m + i)/4](63,48);
    }
    const Word wt_wrd = w[(n*M+m)/WORD_SIZE];

    for (unsigned i = 0; i < WORD_SIZE; ++i) {
      if (wt_wrd[i] > 0)
        sum += in_wrd[i];
      else
        sum -= in_wrd[i];
    }
  }
  return sum;
}

// -----------------------------------------------------------------------
// Internal dense layer
// -----------------------------------------------------------------------
// ML: k, h is the coefficient of BNN!
// ML: the size of in is M/DATA_PER_WORD = M/4 words; the size of out is N/DATA_PER_WORD!
void dense_layer(
    Word data_i[DMEM_WORDS],
    Word data_o[DMEM_O_WORDS],
    unsigned layer_idx,
    const bool inputs_words,
    const Address n_inputs,
    const Address n_outputs,
    Word wt[WT_WORDS],
    Word b[BIAS_WORDS]
) {
  //t_dense.start();
  static Word dmem[3][HID_SIZE/DATA_PER_WORD] = {0}; // ML: sequence: input/output, hid1, hid2

  

  unsigned M = n_inputs;
  unsigned N = n_outputs;

  //ap_uint<1> d_i_idx = dmem_mode;
  //ap_uint<1> d_o_idx = ~dmem_mode;

  static Word in[2*HID_SIZE/DATA_PER_WORD];
  static DATA gate[3][HID_SIZE];  // ML: update gate, reset gate, hidden update gate

  if (layer_idx < 2) {
    LOOP_DMEM_I:
    for (unsigned i = 0; i < M+N; i+= DATA_PER_WORD) {
      if ((i < M) && (layer_idx == 0) && (inputs_words != 0) ) {
        in[i/DATA_PER_WORD] = data_i[i/DATA_PER_WORD];
      }
      else if ((i < M) && (layer_idx == 0) && (inputs_words == 0)) {
        in[i/DATA_PER_WORD] = dmem[0][i/DATA_PER_WORD];
      }
      else if ((i < M) && (layer_idx == 1)) {
        in[i/DATA_PER_WORD] = dmem[1][i/DATA_PER_WORD];
      }
      else if ((i >= M) && (layer_idx == 0)) {
        in[i/DATA_PER_WORD] = dmem[1][(i-M)/DATA_PER_WORD];
      }
      else{
        in[i/DATA_PER_WORD] = dmem[2][(i-M)/DATA_PER_WORD];
      }
    }
  } else {
    LOOP_DMEM_II:
    for (unsigned i = 0; i < M; i+= DATA_PER_WORD) {
      in[i/DATA_PER_WORD] = dmem[2][i/DATA_PER_WORD];
    }
  }
  
  static Word wt_i[WT_WORDS] = {0};
  static Word b_i[BIAS_WORDS] = {0};

  LOOP_WT_I:
  for (unsigned j = 0; j < WT_WORDS; ++j)
    wt_i[j] = wt[j];
  LOOP_B_I:
  for (unsigned j = 0; j < BIAS_WORDS; ++j)
    b_i[j] = b[j];


  if (layer_idx == LAYER_DENSE){
    LOOP_DENSE_O:
    for (unsigned n = 0; n < N; n+=WORD_SIZE) {
      Word out_wrd[WORD_SIZE/DATA_PER_WORD] = {0};
      LOOP_DENSE_I:
      for (unsigned nb = 0; nb < WORD_SIZE; ++nb) {
        DATA sum = dotproduct_m(in, wt_i, M, n+nb);
        out_wrd[nb/DATA_PER_WORD]((nb%DATA_PER_WORD+1)*16-1, (nb%DATA_PER_WORD)*16) = sum(15,0);
      }
      LOOP_DMEM_O:
      for (unsigned i = 0; i < WORD_SIZE / DATA_PER_WORD; ++i){
        data_o[n/DATA_PER_WORD + i] = out_wrd[i];
        dmem[0][n/DATA_PER_WORD + i] = out_wrd[i]; // ML: dont need another data buffer?
      }
      
    }
  } else {
    LOOP_RNN_O:
    for (unsigned n = 0; n < 2*N; n+=WORD_SIZE) {   // ML: compute update gate and reset gate
      LOOP_RNN_I:
      for (unsigned nb = 0; nb < WORD_SIZE; ++nb) {
        DATA sum = dotproduct_m(in, wt_i, M+N, n+nb);
        unsigned gate_idx = (n + nb) / N;
        unsigned gate_off = (n + nb) % N;
        unsigned idx = (n+nb)/DATA_PER_WORD;
        unsigned off = (n+nb)%DATA_PER_WORD;
        DATA temp;
        DATA bias;
        bias(15,0) = b_i[idx]((off+1)*16-1, (off*16));
        gate[gate_idx][gate_off](15,0) = sum(15,0) + bias;
        temp = sigmoid(gate[gate_idx][gate_off]);
        gate[gate_idx][gate_off] = temp;
      }
      
    }

    LOOP_HIDUPDATE:
    for (unsigned n = 2*N; n < 3*N; n+=WORD_SIZE) {
      LOOP_H_O:
      for (unsigned nb = 0; nb < WORD_SIZE; ++nb) {
        LOOP_H_I:
        for (unsigned i = M; i < M+N; i++) {
          unsigned idx = i/DATA_PER_WORD;
          unsigned off = i%DATA_PER_WORD;
          DATA temp;
          temp(15,0) = in[idx]((off+1)*16-1, off*16);
          temp = temp*gate[1][n + nb - 2*N];
          in[idx]((off+1)*16-1, off*16) = temp(15,0); // ML: hid_pre dotproduct by reset gate
        }
        DATA sum = dotproduct_m(in, wt_i, M+N, n+nb);
        unsigned gate_idx = (n + nb) / N;
        unsigned gate_off = (n + nb) % N;
        unsigned idx = (n+nb)/DATA_PER_WORD;
        unsigned off = (n+nb)%DATA_PER_WORD;
        DATA temp;
        DATA bias;
        bias(15,0) = b_i[idx]((off+1)*16-1, (off*16));
        gate[gate_idx][gate_off](15,0) = sum(15,0) + bias;
        temp = tanh(gate[gate_idx][gate_off]);
        gate[gate_idx][gate_off] = temp;
      }
    }
    /*for (unsigned n = 0; n < 2*N; n++) {
      unsigned gate_idx = n / N;
      unsigned gate_off = n % N;
      DATA temp;
      temp = sigmoid(gate[gate_idx][gate_off]);
      gate[gate_idx][gate_off] = temp;
    }*/

    LOOP_DMEM:
    for (unsigned n = 0; n < N; n++) {
      DATA hidden;
      DATA hidden_pre;

      unsigned idx = n / DATA_PER_WORD;
      unsigned offset = n % DATA_PER_WORD;
      hidden_pre(15,0) = dmem[layer_idx+1][idx]((offset+1)*16-1, (offset)*16);
      // ML: new cell state
      hidden = (1-gate[0][n]) * hidden_pre + gate[0][n]*gate[2][n];
      dmem[layer_idx+1][idx]((offset+1)*16-1, offset*16) = hidden(15,0);
    }
  }


  //t_dense.stop();
}

