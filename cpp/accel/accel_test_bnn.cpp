#include <cstddef>
#include <cstdlib>
#include <hls_video.h>

#include "Accel.h"
#include "AccelSchedule.h"
#include "AccelTest.h"
#include "Dense.h"
#include "ZipIO.h"
#include "ParamIO.h"
#include "DataIO.h"
#include "Timer.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    printf ("Give number of character to produce as 1st arg\n");
    return 0;
  }
  const unsigned n_char = std::stoi(argv[1]);

  // print some config numbers
  printf ("* WT_WORDS   = %u\n", WT_WORDS); 
  printf ("* BIAS_WORDS = %u\n", BIAS_WORDS);

  // Load input data
  //printf ("## Loading input data ##\n");
  // ML: hidden state can be initialized by a given string

  // Load parameters
  printf ("## Loading parameters ##\n");
  Params params(get_root_dir() + "/params/rnn_parameters.zip");

  // ---------------------------------------------------------------------
  // allocate and binarize all weights
  // ---------------------------------------------------------------------
  Word* wt[N_LAYERS];   
  Word* b[N_LAYERS];

  for (unsigned l = 0; l < N_LAYERS; ++l) {
    const unsigned M = M_tab[l];
    const unsigned N = N_tab[l];
 
    if (layer_is_rnn(l+1)) {
      wt[l] = new Word[(M+N)*4*N / WORD_SIZE];
      b[l] = new Word[4*N / WORD_SIZE];
    }
    else {
      wt[l] = new Word[M*N / WORD_SIZE];    // ML: RNN layers
      b[l] = new Word[N / WORD_SIZE];
    }
    if (layer_is_rnn(l+1)) {
      for (unsigned w_l = 0; w_l < N_W_LAYERS; ++w_l) {
        // ML: set in_to weight and hid_to weight
        std::cout<<l<<'/'<<w_l<<'\n';
        const float* weights_in = params.float_data(widx_tab[l*N_W_LAYERS*2 + 2*w_l]);
        const float* weights_hid = params.float_data(widx_tab[l*N_W_LAYERS*2 + 2*w_l +1]);
        std::cout<<params.array_size(widx_tab[l*N_W_LAYERS*2 + 2*w_l])<<'\n';
        std::cout<<params.array_size(widx_tab[l*N_W_LAYERS*2 + 2*w_l+1])<<'\n';
        set_rnn_weight_array(wt[l], weights_in, weights_hid, l+1, w_l);
        // ML: set bias
        const float* bias = params.float_data(bidx_tab[l*N_W_LAYERS + w_l]);
        std::cout<<params.array_size(bidx_tab[l*N_W_LAYERS + w_l])<<'\n';
        set_rnn_bias_array(b[l], bias, l+1, w_l);
      }
    } else {

      const float* weights = params.float_data(widx_tab[16]);
      set_dense_weight_array(wt[l], weights, l+1);
      std::cout<<params.array_size(widx_tab[16])<<'\n';
      const float* bias = params.float_data(bidx_tab[8]);
      std::cout<<params.array_size(bidx_tab[8])<<'\n';
      set_dense_bias_array(b[l], bias, l+1);
    }

 
  }

  // ---------------------------------------------------------------------
  // // compute accelerator schedule (divides up weights)
  // ---------------------------------------------------------------------
  AccelSchedule layer_sched[N_LAYERS];    
  for (unsigned l = 0; l < N_LAYERS; ++l) {
    compute_accel_schedule(
        wt[l], b[l],
        M_tab[l], N_tab[l], T_tab[l],
        layer_sched[l], l
    );
  }

  // allocate memories for data i/o for the accelerator
  Word* data_i  = (Word*) MEM_ALLOC( DMEM_WORDS * sizeof(Word) );   // ML: need to be modified!
  Word* data_o  = (Word*) MEM_ALLOC( DMEM_O_WORDS * sizeof(Word) );
  if (!data_i || !data_o) {
    fprintf (stderr, "**** ERROR: Alloc failed in %s\n", __FILE__);
    return (-2);
  }

  unsigned n_errors = 0;

  printf ("## Running RNN for %d characters\n", n_char);

  //--------------------------------------------------------------
  // Run RNN
  //--------------------------------------------------------------

  // ML: load an arbitrary input character [1, 0. 0, ..., 0]
  for (unsigned i = 0; i < VOCAB_SIZE/DATA_PER_WORD; ++i) {
    if (i == 0) {
      data_i[i] = 0;
      DATA start_seed = 1;
      data_i[i](15,0) = start_seed(15,0);
    } else {
      data_i[i] = 0;
    }
  }

  for (unsigned n = 0; n < n_char; ++n) {
   
    //------------------------------------------------------------
    // Execute RNN layers
    //------------------------------------------------------------
    for (unsigned l = 1; l <= 3; ++l) {
      const unsigned M = M_tab[l-1];
      const unsigned N = N_tab[l-1];

      dense_layer(
        data_i, data_o,
        l-1,
        (n==0 && l==1) ? (64/DATA_PER_WORD) : 0,    // input_words
        (l==3) ? (64/DATA_PER_WORD) : 0,
        layer_sched[l-1]
      );  
    }

    //------------------------------------------------------------
    // Execute the prediciton
    //------------------------------------------------------------
    int prediction = 0;
    int max = -512; // ML: may shoulb be less

   
    for (unsigned i = 0; i < VOCAB_SIZE; i++) {
      DATA temp;
      int add = i / DATA_PER_WORD;
      int off = i % DATA_PER_WORD;
      temp(15,0) = data_o[add]((off+1)*16-1,off*16);
      if (temp.to_int() > max) {
        max = temp;
        prediction = i;
      }
    }
      
    

    assert(prediction >= 0 && prediction <= 63);

    std::cout<<vocab[prediction];
    
  }

  /*printf ("\n");
  printf ("Errors: %u (%4.2f%%)\n", n_errors, float(n_errors)*100/n_imgs);
  printf ("\n");
  printf ("Total accel runtime = %10.4f seconds\n", total_time());
  printf ("\n");*/

  MEM_FREE( data_o );
  MEM_FREE( data_i );
  for (unsigned n = 0; n < N_LAYERS; ++n) {
    delete[] wt[n];
    delete[] b[n];
  }
  return 0;
}
