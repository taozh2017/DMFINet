import os

import torch.utils.data
from torchvision.transforms import Compose, Lambda

from data.base_data_loader import BaseDataLoader


def initialize(opt):
    data_path = opt.data_path
#         if opt.dataset_mode == 'unaligned':
#             self.dataset = get3DDataset(data_path)
#         if opt.dataset_mode == 'aligned':
#             print("loading 2d middle data")
#             self.dataset = getMiddle2DDataset(data_path)
#         if opt.dataset_mode == 'unaligned_2D':
#             print("loading 2d middle data")
#             self.dataset = getMiddle2DDataset(data_path)

    if opt.dataset_name == "en2ct":
        from data.EN2CTDataloader import get_loader
        data_loader = get_loader(batchsize=opt.batch_size, shuffle=opt.shuffle, pin_memory=opt.pin_memory, source_modal='en', target_modal='ct',
               img_size=opt.fine_size, img_root=opt.data_path, mode=opt.mode, data_rate=1, num_workers=opt.num_workers, sort=False, argument=opt.argument,
               random=False, re_down=opt.re_down)
        
    if opt.dataset_name == "idb":
        print("using idb dataset")
        from data.IdbDataLoader import get_loader
        data_loader = get_loader(batchsize=opt.batch_size, img_root=opt.data_path, shuffle=opt.shuffle, num_workers=opt.num_workers,
                                    argument=opt.argument, re_down=True, mode=opt.mode,
                                    data_rate=opt.data_rate, img_size=opt.fine_size)    
    if opt.dataset_name == "multi":
        from data.MultiDataLoader import get_loader
        print("aaaaaaaaaaa")
        data_loader = get_loader(batchsize=opt.batch_size, shuffle=opt.shuffle, pin_memory=True, source_modal='t2',
                            target_modal='ct', img_size=opt.fine_size, num_workers=opt.num_workers,
                            img_root=opt.data_path, mode=opt.mode, data_rate=1, argument=opt.argument, random=True, re_down=opt.re_down)
        
    print("the length of data_loader: ", len(data_loader))
    
    return data_loader
