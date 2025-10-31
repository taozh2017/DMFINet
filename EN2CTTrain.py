# coding:utf-8 

import time
from options.train_options import TrainOptions
from models.models import create_model
import logging
import datetime
try:
    from c2net.context import prepare, upload_output
except ImportError as e:
    print(f"Warning: {e}")
    prepare = None
    upload_output = None

# 设置日志格式
logging.basicConfig(filename='EN2CTtrain.log', level=logging.INFO, format='%(asctime)s - %(message)s')
import torch
torch.autograd.set_detect_anomaly(True)
from data.custom_dataset_data_loader import *

if __name__ == "__main__":
    opt = TrainOptions().parse()
    opt.name = opt.name
    if prepare is not None and upload_output is not None:
        # 使用 prepare 和 upload_output
        c2net_context = prepare()
        # 获取代码路径，数据集路径，预训练模型路径，输出路径
        code_path = c2net_context.code_path
        dataset_path = c2net_context.dataset_path
        pretrain_model_path = c2net_context.pretrain_model_path
        you_should_save_here = c2net_context.output_path
    else:
        # 处理没有导入这些模块的情况
        print("The required modules could not be imported.")
    
    print("argument : ", opt.argument)
    if prepare is not None and upload_output is not None:
        if opt.dataset_name == "en2ct":
            data_path = c2net_context.dataset_path+"/"+"dataProcessed/dataProcessed/"
            opt.data_path = data_path
        if opt.dataset_name == "han":
            opt.data_path = c2net_context.code_path + "/myModel/post/"

    print("data_path :  ", opt.data_path)
    data_loader = initialize(opt)
    print(data_loader)
    
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    

    model = create_model(opt)
    total_steps = 0
    model.update_learning_rate(opt.epoch_count-1)
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(data_loader):
            iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            # model.getMSFF()

            if total_steps % 40 == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batch_size
                output_str = f"{i} / {dataset_size},  {epoch} / {opt.niter + opt.niter_decay}\t" + '\t'.join(
                    [f"{k}: {v}" for k, v in errors.items()]
                )

                print(output_str)# 将损失信息写入日志
                logging.info(output_str)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')
        if epoch > 90:
            opt.save_epoch_freq = 1
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch > opt.niter:
            model.update_learning_rate(epoch)


