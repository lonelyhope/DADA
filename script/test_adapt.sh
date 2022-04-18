python ../train_adapt.py \
    --train 0 \
    --save_path experiments_DADA \
    --save_name DADA_Sony_to_Panasonic \
    --save_fix _DADA_Sony_to_Panasonic \
    --save_tst_out Sony_to_Panasonic \
    --s_camera 2 --t_camera 1 \
    --cuda 1 \
    --save_img 1 \
    --resume best.pth
