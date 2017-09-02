#=========================================================================
# opt.tcl
#=========================================================================
set_directive_pipeline dense_layer/LOOP_DMEM_I
set_directive_pipeline dense_layer/LOOP_DMEM_II
set_directive_pipeline dense_layer/LOOP_WT_I
set_directive_pipeline dense_layer/LOOP_B_I
set_directive_pipeline dense_layer/LOOP_DMEM_O

set_directive_loop_tripcount -min 48 -max 64 dense_layer/LOOP_DMEM_I
set_directive_loop_tripcount -min 32 -max 32 dense_layer/LOOP_DMEM_II

#set_directive_unroll dense_layer/LOOP_WT_I
#set_directive_unroll dense_layer/LOOP_B_I

set_directive_pipeline dense_layer/LOOP_DENSE_I
set_directive_loop_tripcount -min 1 -max 2 dense_layer/LOOP_DENSE_O

set_directive_loop_tripcount -min 8 -max 8 dense_layer/LOOP_RNN_O
set_directive_pipeline dense_layer/LOOP_RNN_I

set_directive_loop_tripcount -min 512 -max 512 dense_layer/LOOP_ACT
set_directive_pipeline dense_layer/LOOP_ACT

set_directive_loop_tripcount -min 128 -max 128 dense_layer/LOOP_DMEM
set_directive_pipeline dense_layer/LOOP_DMEM


set_directive_array_partition dense_layer dmem -dim 1 -type complete
set_directive_array_partition dense_layer in -dim 0 -type complete
set_directive_array_partition dense_layer gate -dim 0 -type complete



#=========================================================================
