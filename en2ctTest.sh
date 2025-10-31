python EN2CTTest.py --data_path ../../data/dataProcessed/ --large_num 3 --which_epoch 109  --res_num 3  --name en2ct5 --model cyclegan_2D --which_model_netG mymodel --which_direction AtoB --dataset_mode unaligned_2D --norm instance --fine_size 512 --batch_size 1 --patch_size 8 --mask_ratio 0.0 --base_filter 64 --res_net --mode test/ --dataset_name en2ct --re_down

# python -m  pytorch_fid result/en2ct/output result/en2ct/gt
# python -m  pytorch_fid result/en2ct/reverse result/en2ct/input
# fidelity --kid --input1 result/en2ct/output --input2 result/en2ct/gt --kid-subset-size 193 
# fidelity --kid --input1 result/en2ct/reverse --input2 result/en2ct/input --kid-subset-size 193 