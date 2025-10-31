import logging

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from .transformer_configs import *
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import copy
from .transformer_blocks import Block, Embeddings, np2th
logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, y):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states, y)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        #return encoded, attn_weights
        return encoded


class block(nn.Module):
    def __init__(self,config, input_dim=128, img_size=64,img_size_y=32, input_dim_y=128, patch=True, vis=False):
        super(block, self).__init__()

        self.transformer_encoder = Encoder(config, vis)
        self.config = config
        # Patch embedings
        self.embeddings = Embeddings(config, img_size=img_size, input_dim=input_dim)
        self.embeddings_y = Embeddings(config, img_size=img_size_y, input_dim=input_dim_y)
        self.patch = patch

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            # 因为预训练模型预期输入为14 14 1024, 而我们的输入和他的差距比较大，如果将输入通过卷积改为这种，会带来影响。于是patch_embedding不加载预训练
            # if self.config.name == 'b16':
            #     a = np2th((weights["embedding/kernel"]))
            #     self.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            #     self.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            a = np2th(weights["Transformer/encoder_norm/scale"])
            self.transformer_encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer_encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.embeddings.positional_encoding
            if posemb.size() == posemb_new.size():
                self.embeddings.positional_encoding.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.embeddings.positional_encoding.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.embeddings.positional_encoding.copy_(np2th(posemb))

            # 处理 positional_encoding_y
            posemb_new_y = self.embeddings_y.positional_encoding
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            if posemb.size() == posemb_new_y.size():
                self.embeddings_y.positional_encoding.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new_y.size()[1]:
                posemb = posemb[:, 1:]
                self.embeddings_y.positional_encoding.copy_(posemb)
            else:
                logger.info("load_pretrained for embeddings_y: resized variant: %s to %s" % (
                posemb.size(), posemb_new_y.size()))
                ntok_new_y = posemb_new_y.size(1)
                _, posemb_grid_y = posemb[:, :1], posemb[0, 1:]
                gs_old_y = int(np.sqrt(len(posemb_grid_y)))
                gs_new_y = int(np.sqrt(ntok_new_y))
                print('load_pretrained: grid-size from %s to %s' % (gs_old_y, gs_new_y))
                posemb_grid_y = posemb_grid_y.reshape(gs_old_y, gs_old_y, -1)
                zoom_y = (gs_new_y / gs_old_y, gs_new_y / gs_old_y, 1)
                posemb_grid_y = ndimage.zoom(posemb_grid_y, zoom_y, order=1)  # th2np
                posemb_grid_y = posemb_grid_y.reshape(1, gs_new_y * gs_new_y, -1)
                posemb_y = posemb_grid_y
                self.embeddings_y.positional_encoding.copy_(np2th(posemb_y))

            # Encoder whole
            for bname, block in self.transformer_encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

    def forward(self, x, y):
        if self.patch:
            x = self.embeddings(x)
        y = self.embeddings_y(y)
        x = self.transformer_encoder(x, y)

        return x


#Patch embedings
# self.embeddings = Embeddings(config, img_size=img_size, input_dim=input_dim)

# mymodelName = "mymodelb"
# configs = transformer_configs.CONFIGS[mymodelName]

# a = block(config = configs, input_dim=128, img_size_y=32, input_dim_y=128)
# a.load_from(weights=np.load(configs.pretrained_path))

# b = torch.rand((4, 128, 64, 64))
# d = torch.rand((4, 128, 32, 32))
# c = a(b, d)