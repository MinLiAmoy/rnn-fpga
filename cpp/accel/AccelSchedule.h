#ifndef ACCEL_ACCEL_SCHEDULE_H
#define ACCEL_ACCEL_SCHEDULE_H

#include <vector>
#include "Accel.h"

// Contains all info needed to invoke the accelerator once except
// input/output data and its size which is handled separately
struct AccelInfo {
  Word* wt;
  Word* b;
  unsigned n_inputs;
  unsigned n_outputs;
  ap_uint<3> layer_mode;  // [0]='new layer', [2:1]='rnn1,rnn2,dense'


  AccelInfo() {
    wt = new Word[WT_WORDS];
    b = new Word[BIAS_WORDS];
  }

  ~AccelInfo() {
    delete[] wt;
    delete[] b;
  }
};

typedef std::vector<AccelInfo> AccelSchedule;

void compute_accel_schedule(
    Word* wt,
    Word* b,
    unsigned n_inputs,
    unsigned n_outputs,
    const ap_uint<2> layer_type,  // 0=rnn1, 1=rnn2, 2=dense
    AccelSchedule &schedule,
    unsigned layer_idx
);


void load_weights(Word* wt, Word* wt_o,
                  unsigned o, unsigned n_in, unsigned n_out);

void load_bias(Word* b, Word b_i[], unsigned o, unsigned n_out);

#endif
