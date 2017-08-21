#include "Accel.h"
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
  out = 1/(1+exp(-in));
  return out;
}

DATA tanh(
  const DATA in
  ) {
  DATA out;
  out = (exp(in) - exp(-in)) / (exp(in) + exp(-in));
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
    Word* data_i,
    Word* data_o,
    unsigned layer_idx,
    const Address inputs_words,
    const Address outputs_words,
    AccelSchedule& s 
) {
  //t_dense.start();
  static Word dmem[5][HID_SIZE/DATA_PER_WORD] = {0}; // ML: sequence: input/output, hid1, cell1, hid2, cell2

  

  unsigned M = s[0].n_inputs;
  unsigned N = s[0].n_outputs;

  //ap_uint<1> d_i_idx = dmem_mode;
  //ap_uint<1> d_o_idx = ~dmem_mode;

  Word in[(M+N)/DATA_PER_WORD];
  DATA gate[4][N];  // ML: input, forget, cell(tanh), output

  if (layer_idx < 2) {
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
        in[i/DATA_PER_WORD] = dmem[3][(i-M)/DATA_PER_WORD];
      }
    }
  } else {
    for (unsigned i = 0; i < M; i+= DATA_PER_WORD) {
      in[i/DATA_PER_WORD] = dmem[3][i/DATA_PER_WORD];
    }
  }
  
  static Word* wt_i = (Word*) MEM_ALLOC( WT_WORDS*sizeof(Word));
  static Word* b_i = (Word*) MEM_ALLOC( BIAS_WORDS*sizeof(Word));

  for (unsigned j = 0; j < WT_WORDS; ++j)
    wt_i[j] = s[0].wt[j];
  for (unsigned j = 0; j < BIAS_WORDS; ++j)
    b_i[j] = s[0].b[j];


  if (layer_idx == LAYER_LAST){
    for (unsigned n = 0; n < N; n+=WORD_SIZE) {
      Word out_wrd[WORD_SIZE/DATA_PER_WORD] = {0};
      for (unsigned nb = 0; nb < WORD_SIZE; ++nb) {
        DATA sum = dotproduct_m(in, wt_i, M, n+nb);
        out_wrd[nb/DATA_PER_WORD]((nb%DATA_PER_WORD+1)*16-1, (nb%DATA_PER_WORD)*16) = sum(15,0);
      }
      for (unsigned i = 0; i < WORD_SIZE / DATA_PER_WORD; ++i){
        data_o[n/DATA_PER_WORD + i] = out_wrd[i];
        dmem[0][n/DATA_PER_WORD + i] = out_wrd[i]; // ML: dont need another data buffer?
      }
      
    }
  } else {
    for (unsigned n = 0; n < 4*N; n+=WORD_SIZE) {
      //Word out_wrd[WORD_SIZE/DATA_PER_WORD] = {0};
      for (unsigned nb = 0; nb < WORD_SIZE; ++nb) {
        DATA sum = dotproduct_m(in, wt_i, M+N, n+nb);
        //out_wrd[nb/DATA_PER_WORD]((nb%DATA_PER_WORD+1)*16-1, (nb%DATA_PER_WORD)*16-1) = sum(15,0);
        unsigned gate_idx = (n + nb) / N;
        unsigned gate_off = (n + nb) % N;
        unsigned idx = (n+nb)/DATA_PER_WORD;
        unsigned off = (n+nb)%DATA_PER_WORD;
        DATA bias;
        bias(15,0) = b_i[idx]((off+1)*16-1, (off*16));
        gate[gate_idx][gate_off](15,0) = sum(15,0) + bias;
      }
      
    }

    for (unsigned n = 0; n < 4*N; n++) {
      unsigned gate_idx = n / N;
      unsigned gate_off = n % N;
      DATA temp;

      if (gate_idx != 2) {
        temp = sigmoid(gate[gate_idx][gate_off]);
        gate[gate_idx][gate_off] = temp;
      }
      else {
        temp = tanh(gate[gate_idx][gate_off]);
        gate[gate_idx][gate_off] = temp;
      }
    }

    for (unsigned n = 0; n < N; n++) {
      DATA cell;
      DATA cell_pre;
      DATA hidden;
      unsigned idx = n / DATA_PER_WORD;
      unsigned offset = n % DATA_PER_WORD;
      cell_pre(15,0) = dmem[(layer_idx+1)*2][idx]((offset+1)*16-1, (offset)*16);
      // ML: new cell state
      cell = gate[1][n] * cell_pre + gate[0][n]*gate[2][n];
      hidden = gate[3][n] * tanh(cell);
      dmem[(layer_idx+1)*2][idx]((offset+1)*16-1, offset*16) = cell(15,0);
      dmem[(layer_idx+1)*2 - 1][idx]((offset+1)*16-1, offset*16) = hidden(15,0);
    }


  }


  //t_dense.stop();
}

