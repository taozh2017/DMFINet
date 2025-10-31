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
    is_train = None
    if opt.mode == "train/":
        is_train = True
    else:
        is_train = False

    if opt.dataset_name == "en2ct":
        from data.EN2CTDataloader import get_loader
        data_loader = get_loader(batchsize=opt.batch_size, shuffle=opt.shuffle, pin_memory=opt.pin_memory, source_modal='en', target_modal='ct',
               img_size=opt.fine_size, img_root=opt.data_path, model=opt.mode, data_rate=1, num_workers=opt.num_workers, sort=False, argument=opt.argument,
               random=False, re_augment=False, re_down=True)
    elif opt.dataset_name == "han":
        print("using han dataset")
        from data.HanDataLoader import get_loader
        data_loader = get_loader(batch_size=opt.batch_size, data_path=opt.data_path, shuffle=opt.shuffle, num_workers=opt.num_workers, is_train=is_train, argument=opt.argument, re_augment=False, re_down=True, data_rate = opt.data_rate)
    print("the length of data_loader: ", len(data_loader))
    
    return data_loader
