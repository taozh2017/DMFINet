import os
import torch
from c2net.context import prepare, upload_output

class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        c2net_context = prepare()
    # 获取代码路径，数据集路径，预训练模型路径，输出路径
        code_path = c2net_context.code_path
        dataset_path = c2net_context.dataset_path
        pretrain_model_path = c2net_context.pretrain_model_path
        you_should_save_here = c2net_context.output_path
        self.save_dir = you_should_save_here
        self.load_path = "result/"
       # self.load_path = pretrain_model_path+"/"+"CycleGanMr2Ct3d_model_tldd/"
        self.save_dir = "result/"
        os.makedirs(self.save_dir, exist_ok=True)
        print(self.load_path)
        

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            #network.cuda(device_id=gpu_ids[0])
            network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.load_path, save_filename)
        print(save_path)
        #save_path = save_filename
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate():
        pass
