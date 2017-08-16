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

  /*const unsigned lconv  = 6;  // last conv
  const unsigned ldense = 8;  // last dense
  const bool DENSE_LAYER_CPU = getenv("BNN_DENSE_LAYER_CPU") != NULL;   // ML: can be defined above
  const bool LAST_LAYER_CPU = getenv("BNN_LAST_LAYER_CPU") != NULL;
  if (DENSE_LAYER_CPU)
    printf ("## Dense layer CPU is turned on ##\n");
  if (LAST_LAYER_CPU)
    printf ("## Last layer CPU is turned on ##\n");*/

  // print some config numbers
  printf ("* WT_WORDS   = %u\n", WT_WORDS); 
  printf ("* KH_WORDS   = %u\n", KH_WORDS);     // ML: *

  // Load input data
  /*printf ("## Loading input data ##\n");
  Cifar10TestInputs X(n_imgs);    
  Cifar10TestLabels y(n_imgs);*/

  // Load parameters
  printf ("## Loading parameters ##\n");
  Params params(get_root_dir() + "/params/rnn_parameters.zip");

  // ---------------------------------------------------------------------
  // allocate and binarize all weights
  // ---------------------------------------------------------------------
  Word* wt[N_LAYERS];   // *ML: need to be modified
  //Word* b[N_LAYERS];
  //Word* kh[N_LAYERS];   
  for (unsigned l = 0; l < N_LAYERS; ++l) {
    const unsigned M = M_tab[l];
    const unsigned N = N_tab[l];
    /*if (layer_is_conv(l+1))
      wt[l] = new Word[WTS_TO_WORDS(M*N)];    // ML: roughly a word contains 7 binarized 3*3 conv param.
    else*/
    if (layer_is_rnn(l+1))
      wt[l] = new Word[4N*(M+N)/WORD_SIZE];
    else
      wt[l] = new Word[M*N/WORD_SIZE];    // ML: RNN layers

    if (layer_is_rnn(l+1)) {
      for (unsigned w_l = 0; w_l < N_W_LAYERS; ++w_l) {
        const float* weights = params.float_data(widx_tab[l*N_W_LAYERS+w_l]);
        set_weight_array_rnn(wt[l], weights, l+1, w_l);
      }
    } else {
      const float* weights = params.float_data(widx_tab[16]);
      set_weight_array(wt[l], weights, l+1);    // 
    }

    /*kh[l] = new Word[N/KH_PER_WORD * sizeof(Word)];   // ML: * why op *sizeof(Word)? bug! when it comes to last layer, the size is N/2 but not N
    const float* k = params.float_data(kidx_tab[l]);
    const float* h = params.float_data(hidx_tab[l]);
    set_bnorm_array(kh[l], k, h, l+1);*/
  }

  // ---------------------------------------------------------------------
  // // compute accelerator schedule (divides up weights)
  // ---------------------------------------------------------------------
  AccelSchedule layer_sched[N_LAYERS];
  for (unsigned l = 0; l < N_LAYERS; ++l) {
    compute_accel_schedule(
        wt[l], 
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

  printf ("## Running BNN for %d characters\n", n_char);

  //--------------------------------------------------------------
  // Run BNN
  //--------------------------------------------------------------

  // ML: load an arbitrary input character [1, 0. 0, ..., 0]
  for (unsigned i = 0; i < 64/DATA_PER_WORD; ++i) {
    if (i == 0) {
      data_i[i] = 0;
      DATA start_seed = 1;
      data_i[i](15,0) = start_seed(15,0);
    } else {
      data[i] = 0;
    }
  }

  for (unsigned n = 0; n < n_char; ++n) {
    /*float* data = X.data + n*3*32*32;
    binarize_input_images(data_i, data, 32);*/
    
    // ML:have to define a data_i!

    //------------------------------------------------------------
    // Execute conv layers
    //------------------------------------------------------------
    /*for (unsigned l = 1; l <= lconv; ++l) {
      const unsigned M = M_tab[l-1];
      const unsigned N = N_tab[l-1];
      const unsigned S = S_tab[l-1];
      unsigned input_words = (l==1) ? S*S : M*S*S/WORD_SIZE;
      unsigned output_words = (pool_tab[l-1]) ? N*S*S/WORD_SIZE/4 : N*S*S/WORD_SIZE;

      run_accel_schedule(
          data_i, data_o,
          l-1,        // layer_idx
          (l==1) ? input_words : 0,
          (l==lconv && DENSE_LAYER_CPU) ? output_words : 0,
          l % 2,      // mem_mode
          layer_sched[l-1]
      );
    }*/

    //------------------------------------------------------------
    // Execute dense layers
    //------------------------------------------------------------
    for (unsigned l = 1; l <= 3; ++l) {
      const unsigned M = M_tab[l-1];
      const unsigned N = N_tab[l-1];

    dense_layer_cpu(
      data_i, data_o,
      l-1,
      (n==1) ? (64/DATA_PER_WORD) : 0,    // input_words
      (l==3) ? (64/DATA_PER_WORD) : 0,
      //l % 2,
      layer_sched[l-1]
    );  
    
    }

    //------------------------------------------------------------
    // Execute last layer
    //------------------------------------------------------------
    int prediction = -1;
    /*if (DENSE_LAYER_CPU || LAST_LAYER_CPU) {
      prediction = last_layer_cpu(
          wt[ldense],
          params.float_data(kidx_tab[ldense]),
          params.float_data(hidx_tab[ldense]),
          data_o,
          M_tab[ldense], N_tab[ldense]
      );
    } else {
      run_accel_schedule(
          data_i, data_o,
          ldense,
          0, 1,
          1,
          layer_sched[ldense]
      );*/
      ap_int<8> p = 0;
      p(7,0) = data_o[0](7,0);
      prediction = p.to_int();
    

    //assert(prediction >= 0 && prediction <= 9);
    int label = y.data[n];

    printf ("  Pred/Label:\t%2u/%2d\t[%s]\n", prediction, label,
        ((prediction==label)?" OK ":"FAIL"));

    n_errors += (prediction!=label);
  }

  printf ("\n");
  printf ("Errors: %u (%4.2f%%)\n", n_errors, float(n_errors)*100/n_imgs);
  printf ("\n");
  printf ("Total accel runtime = %10.4f seconds\n", total_time());
  printf ("\n");

  MEM_FREE( data_o );
  MEM_FREE( data_i );
  for (unsigned n = 0; n < N_LAYERS; ++n) {
    delete[] wt[n];
    delete[] kh[n];
  }
  return 0;
}
