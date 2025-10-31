python EN2CTTrain.py --gpu_ids 1 --data_path ../../data/dataProcessed/ --model cyclegan_2D --which_model_netG mymodel --which_direction AtoB --lambda_A 10 --lambda_B 10 --identity 0.5 --pool_size 50 --dataset_mode unaligned_2D --norm instance --fine_size 512 --batch_size 1 --input_nc 1 --output_nc 1 --save_epoch_freq 1 --niter 50 --niter_decay 50 --patch_size 8 --mask_ratio 0.3 --base_filter 64 --res_net --no_lsgan --argument --dataset_name en2ct --mode train/ --re_down --large_num 3 --res_num 3 --re_loss_weight 1 --name en2ctTest
# --continue_train --which_epoch 20 --epoch_count 21

# --res_net 启动resnet
# --norm batch  / instance  决定是否使用bias
# --no_lsgan 是否使用lsgan 
# argument 是否训练数据增强
