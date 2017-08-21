#include "AccelSchedule.h"
#include "AccelTest.h"
#include "Timer.h"

static Timer timers[N_LAYERS] = {
  "xl-FC",
  "xl-RNN2",
  "xl-RNN1",
};

// -----------------------------------------------------------------------
// Each layer may need multiple invocations of the accelerator due to
// limited on-chip storage of weights.
//
// This function computes the number of invocations needed and splits
// the weights for each invocation.
//
// We make the following assumptions now:
// 1. Only 1 output image per invocation
// 2. wt_mem is large enough to hold the weights for at least 1 image
// -----------------------------------------------------------------------
void compute_accel_schedule(
    Word* wt,
    Word* b,   
    unsigned n_inputs,
    unsigned n_outputs,
    const ap_uint<2> layer_type,  // 0=rnn1, 1=rnn2, 2=dense
    AccelSchedule &schedule,
    unsigned layer_idx
) {
  assert (wt != NULL);
  assert (b  != NULL);

  ap_uint<3> layer_mode = 0;
  layer_mode(2,1) = layer_type(1,0);

  unsigned idx = 0;

  schedule.resize(1);

  layer_mode[0] = 1;

  // add a new invocation to the schedule
  schedule[idx].n_inputs = n_inputs;    // ML: n_input has been modified
  schedule[idx].n_outputs = n_outputs;
  schedule[idx].layer_mode = layer_mode;


  unsigned o = 0;  // ML: we assume there is no image batch

  Word* wt_i = schedule[idx].wt;
  if (layer_type < 2) {
   load_weights(wt, wt_i, o, n_inputs+n_outputs, 3*n_outputs);    // ML: the weights are loaded on the wt_i
  }
  else {
   load_weights(wt, wt_i, o, n_inputs, n_outputs);
  }

  Word* b_i = schedule[idx].b;
  if (layer_type < 2) {
    load_bias(b, b_i, o, 3*n_outputs);
  } else {
    load_bias(b, b_i, o, n_outputs);
  }
}



// -----------------------------------------------------------------------
// load n_in*n_out single bit weights into accelerator
// o is which output bit we are starting from
// -----------------------------------------------------------------------
void load_weights(Word* wt, Word* wt_o,
                      unsigned o, unsigned n_in, unsigned n_out
) {
  assert(n_in % WORD_SIZE == 0);
  // load in Word-sized chunks
  for (unsigned i = 0; i < n_in*n_out/WORD_SIZE; ++i) {
    wt_o[i] = wt[o*n_in/WORD_SIZE + i];
  }
}

// -----------------------------------------------------------------------
// load n_out sets of kh params into accelerator
// -----------------------------------------------------------------------
void load_bias(Word* b, Word b_i[], unsigned o, unsigned n_out) {
  for (unsigned i = 0; i < n_out / WORD_SIZE; ++i) {
    b_i[i] = b[o + i];
  }
}


