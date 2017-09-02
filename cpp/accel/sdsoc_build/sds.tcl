set rootdir $::env(CRAFT_BNN_ROOT)
source $rootdir/cpp/accel/opt.tcl

set_directive_interface -mode ap_fifo "dense_layer" s[0].wt
set_directive_interface -mode ap_fifo "dense_layer" s[0].b
set_directive_interface -mode ap_fifo "dense_layer" data_i
set_directive_interface -mode ap_fifo "dense_layer" data_o

#set_directive_interface -mode bram "top" wt_i
#set_directive_interface -mode bram "top" kh_i
#set_directive_resource -core RAM_1P "top" wt_i
#set_directive_resource -core RAM_1P "top" kh_i
