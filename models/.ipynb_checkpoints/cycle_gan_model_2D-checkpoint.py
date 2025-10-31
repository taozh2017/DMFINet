import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from util.networkUtil import getDownAndRe
from util.testUtil import *
from .base_model import BaseModel
from . import networks
import sys
import nibabel as nib
# from . import VGGLoss
from torchvision.utils import save_image


class CycleGANModel_2D(BaseModel):
    def name(self):
        return 'CycleGANModel_2D'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batch_size
        size = opt.fine_size
        # define tensors
        self.input_A = self.Tensor(opt.batch_size, opt.input_nc,
                                   opt.fine_size, opt.fine_size)
        self.input_B = self.Tensor(opt.batch_size, opt.output_nc,
                                   opt.fine_size, opt.fine_size)
        print("inputA size is : ", self.input_A.shape)
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        # self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
        #                                 opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)
        self.netG_A = networks.define_G(opt.patch_size, opt.mask_ratio, opt.base_filter, opt.which_model_netG, opt.norm,
                                        self.gpu_ids, opt.res_net, opt.pretrained_path)
        self.netG_B = networks.define_G(opt.patch_size, opt.mask_ratio, opt.base_filter, opt.which_model_netG, opt.norm,
                                        self.gpu_ids, opt.res_net, opt.pretrained_path)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # 如果no_lsgan为True，则使用sigmoid,但是不使用LSGAN； 如果为false，则不用sigmoid作为鉴别器最后一层，相当于使用LSGAN
            print("nolsgan value  -> ：", use_sigmoid)
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            print("continue training!!!!")
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            self.netG_A.eval()
            self.netG_B.eval()
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMask = torch.nn.L1Loss()
            self.sobelloss = networks.SobelLoss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        #         networks.print_network(self.netG_A)
        #         networks.print_network(self.netG_B)
        #         if self.isTrain:
        #             networks.print_network(self.netD_A)
        #             networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        #         import pdb
        #         pdb.set_trace()
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.A_down_image = input['A_down_image']
        self.B_down_image = input['B_down_image']
        self.A_re_down_image = input['A_re_down_image']
        self.B_re_down_image = input['B_re_down_image']
        self.mask_A = input['mask_A']
        self.mask_B = input['mask_B']
        # print("input shape ", self.input_A.shape)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.image_paths = "checkpoints/image_saved/"
        self.name = input['name']

    def forward(self):
        self.real_A = Variable(self.input_A).cuda()
        self.real_B = Variable(self.input_B).cuda()
        self.A_down_image = Variable(self.A_down_image).cuda()
        self.B_down_image = Variable(self.B_down_image).cuda()
        self.A_re_down_image = Variable(self.A_re_down_image).cuda()
        self.B_re_down_image = Variable(self.B_re_down_image).cuda()
        self.mask_A = Variable(self.mask_A).cuda()
        self.mask_B = Variable(self.mask_B).cuda()

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        re_loss_weight = self.opt.re_loss_weight
        lambda_mask = self.opt.lambda_mask
        lambda_shape = self.opt.lambda_shape
        lambda_sobel = self.opt.lambda_sobel

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            _, _, self.idt_A, _ = self.netG_A.forward(self.real_B, self.B_down_image, self.B_re_down_image)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            _, _, self.idt_B, _ = self.netG_B.forward(self.real_A, self.A_down_image, self.B_re_down_image)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        # D_A(G_A(A))   self开头的是目前使用的
        mask_image, A_re_result, self.fake_B, self.fake_B_attentions = self.netG_A(self.real_A, self.A_down_image, self.A_re_down_image)
        pred_fake = self.netD_A.forward(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake, True)
        # D_B(G_B(B))
        mask_image, B_re_result, self.fake_A, self.fake_A_attentions = self.netG_B(self.real_B, self.B_down_image, self.B_re_down_image)
        pred_fake = self.netD_B.forward(self.fake_A)
        self.loss_G_B = self.criterionGAN(pred_fake, True)
        # Forward cycle loss
        self.fake_B_down_image, self.fake_B_re_down_image = getDownAndRe(self.fake_B, re_down=True, re_augment=False)
        # print(self.fake_B_re_down_image.shape)
        mask_image, cyc_B_re_result, self.rec_A, self.rec_A_attentions = self.netG_B(self.fake_B, self.fake_B_down_image, self.fake_B_re_down_image)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.fake_A_down_image, self.fake_A_re_down_image = getDownAndRe(self.fake_A, re_down=True, re_augment=False)
        mask_image, cyc_A_re_result, self.rec_B, self.rec_B_attentions = self.netG_A(self.fake_A, self.fake_A_down_image, self.fake_A_re_down_image)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss  identity B通过A到B的生成器结果进行损失

        # if lambda_mask > 0:
        #     self.loss_maskB = self.criterionMask(self.fake_B_attentions[-1], self.mask_A)
        #     self.loss_maskA = self.criterionMask(self.fake_A_attentions[-1], self.mask_B)
        #     self.loss_mask = lambda_mask * (self.loss_maskA + self.loss_maskB)
        # else:
        #     self.loss_maskA = self.loss_maskB = 0
        #     self.loss_mask = 0

        # self.loss_sobelB = lambda_sobel * self.sobelloss(self.fake_B, self.real_A)
        # self.loss_sobelA = lambda_sobel * self.sobelloss(self.fake_A, self.real_B)
        # self.loss_RB = lambda_sobel * self.sobelloss(self.rec_A, self.fake_B)
        # self.loss_RA = lambda_sobel * self.sobelloss(self.rec_B, self.fake_A)
        # self.loss_sobel = self.loss_sobelB + self.loss_sobelA + self.loss_RB + self.loss_RA

        # self.loss_shapeB = lambda_shape * self.criterionMask(self.fake_B_attentions[-1], self.rec_A_attentions[-1])
        # self.loss_shapeA = lambda_shape * self.criterionMask(self.fake_A_attentions[-1], self.rec_B_attentions[-1])
        # self.loss_shape = self.loss_shapeA + self.loss_shapeB

        self.re_loss = self.criterionL1(A_re_result, self.A_re_down_image) + self.criterionL1(B_re_result, self.B_re_down_image)
        self.cycle_re_loss = self.criterionL1(self.fake_B_re_down_image, cyc_B_re_result) + self.criterionL1(self.fake_A_re_down_image, cyc_A_re_result)
        self.re_loss_total = self.re_loss + self.cycle_re_loss
        self.loss_G = (self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + re_loss_weight * self.re_loss_total)
        #  self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        D_A = self.loss_D_A.item()
        G_A = self.loss_G_A.item()
        Cyc_A = self.loss_cycle_A.item()
        D_B = self.loss_D_B.item()
        G_B = self.loss_G_B.item()
        Cyc_B = self.loss_cycle_B.item()
        Re_loss = self.re_loss.item()
#         mask_loss = self.loss_mask.item()
#         shape_loss = self.loss_shape.item()
        # sobel_loss = self.loss_sobel.item()
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.item()
            idt_B = self.loss_idt_B.item()
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B),
                                ('LR', self.old_lr),
                                ('re_loss', Re_loss)])
        else:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B),
                                ('LR', self.old_lr),
                                ('re_loss', Re_loss)])
#                                 ('shape_loss', shape_loss),
#                                 ('mask_loss', mask_loss)])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        rec_A = util.tensor2im(self.rec_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        rec_B = util.tensor2im(self.rec_B.data)
        if self.opt.identity > 0.0:
            idt_A = util.tensor2im(self.idt_A.data)
            idt_B = util.tensor2im(self.idt_B.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def update_learning_rate(self, epoch):
        lrd = self.opt.lr / self.opt.niter_decay
        de_epoch = epoch - self.opt.niter
        print(de_epoch)
        lr_de = 0
        if de_epoch > 0:
            lr_de = de_epoch * lrd
        # lr = self.old_lr - lr_de
        lr = self.opt.lr - lr_de
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def test(self):
        print("testing ")
        with torch.no_grad():
            self.real_A = Variable(self.input_A).cuda()
            self.real_B = Variable(self.input_B).cuda()
            self.A_down_image = Variable(self.A_down_image).cuda()
            self.B_down_image = Variable(self.B_down_image).cuda()
            self.A_re_down_image = Variable(self.A_re_down_image).cuda()
            self.B_re_down_image = Variable(self.B_re_down_image).cuda()

            mask_image, re_result, self.fake_B, self.fake_B_attentions = self.netG_A(self.real_A, self.A_down_image, self.A_re_down_image)
            mask_image, re_result, self.fake_A, self.fake_A_attentions = self.netG_B(self.real_B, self.B_down_image, self.B_re_down_image)

    def saveImages(self, result_dir, i):
        origin_path = result_dir + "input/"
        output_path = result_dir + "output/"
        gt_path = result_dir + "gt/"
        os.makedirs(origin_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(gt_path, exist_ok=True)
        real_A = (self.real_A + 1) / 2
        real_B = (self.real_B + 1) / 2
        fake_B = (self.fake_B + 1) / 2

        real_A = torch.rot90(real_A, k=1, dims=(2, 3))
        real_A = torch.flip(real_A, [3])
        real_B = torch.rot90(real_B, k=1, dims=(2, 3))
        real_B = torch.flip(real_B, [3])
        fake_B = torch.rot90(fake_B, k=1, dims=(2, 3))
        fake_B = torch.flip(fake_B, [3])

        print(real_A.min())
        print(real_A.max())
        index = self.name[0]

        save_image(real_A, origin_path + str(index) + 'input.png', normalize=False)
        save_image(fake_B, output_path + str(index) + 'output.png', normalize=False)
        save_image(real_B, gt_path + str(index) + 'gt.png', normalize=False)

    def calculateMetrics(self):
        pred, gt = self.fake_B.cpu().detach().numpy().squeeze(), self.real_B.cpu().detach().numpy().squeeze()  # [-1 1]
        pred, gt = (pred + 1) / 2, (gt + 1) / 2  # to 0-1
        a = ssim(pred, gt, data_range=1)
        b = psnr(pred, gt)
        c = nmse(pred, gt)

        return [a, b, c]
    
    def saveImagesHan(self, result_dir, i):
        origin_path = result_dir + "input/"
        output_path = result_dir + "output/"
        gt_path = result_dir + "gt/"
        os.makedirs(origin_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(gt_path, exist_ok=True)
        real_A = (self.real_A + 1) / 2
        real_B = (self.real_B + 1) / 2
        fake_B = (self.fake_B + 1) / 2
        index = self.name[0]

        save_image(real_A, origin_path + str(index) + 'input.png', normalize=False)
        save_image(fake_B, output_path + str(index) + 'output.png', normalize=False)
        save_image(real_B, gt_path + str(index) + 'gt.png', normalize=False)
