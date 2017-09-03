set rootdir $::env(CRAFT_BNN_ROOT)
source $rootdir/cpp/accel/opt.tcl

set_directive_interface -mode ap_fifo "dense_layer" wt
set_directive_interface -mode ap_fifo "dense_layer" b
set_directive_interface -mode ap_fifo "dense_layer" dmem_i
set_directive_interface -mode ap_fifo "dense_layer" dmem_o

#set_directive_interface -mode bram "top" wt_i
#set_directive_interface -mode bram "top" kh_i
#set_directive_resource -core RAM_1P "top" wt_i
#set_directive_resource -core RAM_1P "top" kh_i
