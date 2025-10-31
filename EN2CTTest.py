import time
import os
from options.test_options import TestOptions
from models.models import create_model
try:
    from c2net.context import prepare, upload_output
except ImportError as e:
    print(f"Warning: {e}")
    prepare = None
    upload_output = None
from data.EN2CTDataloader import get_loader
import numpy as np
from data.custom_dataset_data_loader import *

opt = TestOptions().parse()
opt.name = opt.name
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.shuffle = False  # no shuffle
opt.no_flip = True  # no flip
opt.num_workers = 8
opt.pin_memory = True
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

print("how_many image will be processed: ", opt.how_many)
if prepare is not None and upload_output is not None:
    if opt.dataset_name == "en2ct":
        data_path = c2net_context.dataset_path+"/"+"dataProcessed/dataProcessed/"
        opt.data_path = data_path
    if opt.dataset_name == "han":
        opt.data_path = c2net_context.code_path + "/myModel/post/"
        
data_loader = initialize(opt)

dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
print(opt.model)
result_dir = "result/"
os.makedirs(result_dir, exist_ok=True)
SSIM, NMSE, PSNR = [], [], []
rSSIM, rNMSE, rPSNR = [], [], []
for i, data in enumerate(data_loader):
    print(i)
    if i >= opt.how_many:
        break

    if opt.dataset_name == "en2ct":
        result_dir = "result/en2ct/"
        model.set_input(data)
        model.test()
        model.saveImages(result_dir, i)
        metrics = model.calculateMetrics()
        SSIM.append(metrics[0])
        PSNR.append(metrics[1])
        NMSE.append(metrics[2])
        # model.getData()
        model.getMSFF()

    if opt.dataset_name == "idb":
        result_dir = "result/idb/"
        model.set_input(data)
        model.test()
        model.saveImagesHan(result_dir, i)
        metrics = model.calculateMetrics()
        SSIM.append(metrics[0])
        PSNR.append(metrics[1])
        NMSE.append(metrics[2])

        metrics = model.rcalculateMetrics()
        rSSIM.append(metrics[0])
        rPSNR.append(metrics[1])
        rNMSE.append(metrics[2])

        
    if opt.dataset_name == "multi":
        result_dir = "result/multi/"
        model.set_input(data)
        model.test()
        model.saveImagesHan(result_dir, i)
#        model.getData()
        metrics = model.calculateMetrics()
        SSIM.append(metrics[0])
        PSNR.append(metrics[1])
        NMSE.append(metrics[2])
        if i == 6:
            model.getMSFF()
        metrics = model.rcalculateMetrics()
        rSSIM.append(metrics[0])
        rPSNR.append(metrics[1])
        rNMSE.append(metrics[2])

        
print("A2B")
print("PSNR mean:", np.mean(PSNR), "PSNR std:", np.std(PSNR))
print("NMSE mean:", np.mean(NMSE), "NMSE std:", np.std(NMSE))
print("SSIM mean:", np.mean(SSIM), "SSIM std:", np.std(SSIM))
print("reverse")
print("PSNR mean:", np.mean(rPSNR), "PSNR std:", np.std(rPSNR))
print("NMSE mean:", np.mean(rNMSE), "NMSE std:", np.std(rNMSE))
print("SSIM mean:", np.mean(rSSIM), "SSIM std:", np.std(rSSIM))

# webpage.save()
