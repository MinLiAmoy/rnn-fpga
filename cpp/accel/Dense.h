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

#pragma SDS data copy(data_i[0:input_words], data_o[0:output_words])
#pragma SDS data access_pattern(data_i:SEQUENTIAL, data_o:SEQUENTIAL)
#pragma SDS data mem_attribute(data_i:PHYSICAL_CONTIGUOUS, data_o:PHYSICAL_CONTIGUOUS)
#pragma SDS data data_mover(data_i:AXIDMA_SIMPLE, data_o:AXIDMA_SIMPLE)
#pragma SDS data access_pattern(s[0].wt:SEQUENTIAL, s[0].b:SEQUENTIAL)
#pragma SDS data mem_attribute(s[0].wt:PHYSICAL_CONTIGUOUS, s[0].b:PHYSICAL_CONTIGUOUS)
#pragma SDS data data_mover(s[0].wt:AXIDMA_SIMPLE, s[0].b:AXIDMA_SIMPLE)
void dense_layer(
    const Word* data_i,
    Word* data_o,
    unsigned layer_idx,
    const Address inputs_words,
    const Address outputs_words,
    AccelSchedule& s 
);


#endif
