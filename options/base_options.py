import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--data_path', default="data/data/", help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--fine_size', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--depth_size', type=int, default=32, help='depth for 3d images')
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet_6blocks', help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default="0,1", help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='mr2ct', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--model', type=str, default='cycle_gan',
                                 help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
        self.parser.add_argument('--shuffle', default=True, type=bool, help='shuffle or not')
        self.parser.add_argument('--num_workers', default=16, type=int, help='num_workers')
        self.parser.add_argument('--pin_memory', default=True, type=bool, help='pin_memory')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

        # my model part
        self.parser.add_argument('--patch_size', type=int, default=8, help='mae part patch_size')
        self.parser.add_argument('--mask_ratio', type=float, default=0.2, help='mae part mask_ratio')
        self.parser.add_argument('--base_filter', type=int, default=16, help='the base channel number')
        self.parser.add_argument('--re_loss_weight', type=float, default=1.0, help='the re_loss weight of inner model')
        self.parser.add_argument('--res_net', action='store_true', help='using res_net block')
        self.parser.add_argument('--pretrained_path', type=str, default="/tmp/pretrainmodel/myModel_model_tran/R50+ViT-B_16.npz", help='pretrained_path')
        self.parser.add_argument('--lambda_mask', type=float, default=1.0, help='attention mask')
        self.parser.add_argument('--lambda_shape', type=float, default=0.5, help='shape coffecient')
        self.parser.add_argument('--lambda_sobel', type=float, default=0.25, help='sobel coeff')
        
        
        # dataset
        self.parser.add_argument('--dataset_name', type=str, default='en2ct', help='en2ct or han or')
        self.parser.add_argument('--mode', type=str, default='train/', help='train/ or test/')
        self.parser.add_argument('--argument', action="store_true", help='argument or not')
        self.parser.add_argument('--data_rate', type=float, default=1, help='data_rate of data')
        self.parser.add_argument('--re_down', action="store_true", help='argument or not')
        self.parser.add_argument('--res_num', type=int, default=6, help='number of res_bolock')
        self.parser.add_argument('--large_num', type=int, default=1, help='number of large_conv block')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
