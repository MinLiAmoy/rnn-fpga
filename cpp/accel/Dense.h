#ifndef ACCEL_DENSE_H
#define ACCEL_DENSE_H

#include <cstddef>
#include <hls_math.h>
#include "Debug.h"
#include "Typedefs.h"
#include "Accel.h"
#include "AccelSchedule.h"

/*void dense_layer_cpu(
    const Word* w,
    const float* k_data,
    const float* h_data,
    const Word* data_i,
    Word* data_o,
    const unsigned M,
    const unsigned N
);*/

#pragma SDS data copy(data_i[0:16], data_o[0:16])
#pragma SDS data access_pattern(data_i:SEQUENTIAL, data_o:SEQUENTIAL)
#pragma SDS data mem_attribute(data_i:PHYSICAL_CONTIGUOUS, data_o:PHYSICAL_CONTIGUOUS)
#pragma SDS data data_mover(data_i:AXIDMA_SIMPLE, data_o:AXIDMA_SIMPLE)
#pragma SDS data access_pattern(wt:SEQUENTIAL, b:SEQUENTIAL)
#pragma SDS data mem_attribute(wt:PHYSICAL_CONTIGUOUS, b:PHYSICAL_CONTIGUOUS)
#pragma SDS data data_mover(wt:AXIDMA_SIMPLE, b:AXIDMA_SIMPLE)
void dense_layer(
    Word data_i[DMEM_WORDS],
    Word data_o[DMEM_O_WORDS],
    unsigned layer_idx,
    const bool inputs_words,
    const Address n_inputs,
    const Address n_outputs,
    Word wt[WT_WORDS],
    Word b[BIAS_WORDS]
);


#endif
