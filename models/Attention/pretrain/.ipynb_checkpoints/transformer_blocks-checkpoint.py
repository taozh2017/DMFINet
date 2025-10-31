import math

import torch.nn as nn
from torch.nn.modules.utils import _pair
from . import transformer_configs
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size##paraphrase

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, y):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(y)
        mixed_value_layer = self.value(y)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.attention_norm_y = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x, y):
        h = x
        x = self.attention_norm(x)
        y = self.attention_norm_y(y)
        x, weights = self.attn(x, y)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}/"
        with torch.no_grad():
            # query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            # key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            # value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            # out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            a = np2th(weights[(ROOT  + ATTENTION_Q + "/kernel")])
            query_weight = np2th(weights[(ROOT  + ATTENTION_Q + "/kernel")]).view(self.hidden_size,
                                                                                       self.hidden_size).t()
            key_weight = np2th(weights[(ROOT  + ATTENTION_K + "/kernel")]).view(self.hidden_size,
                                                                                     self.hidden_size).t()
            value_weight = np2th(weights[(ROOT  + ATTENTION_V + "/kernel")]).view(self.hidden_size,
                                                                                       self.hidden_size).t()
            out_weight = np2th(weights[(ROOT + ATTENTION_OUT + "/kernel")]).view(self.hidden_size,
                                                                                       self.hidden_size).t()

            # query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            # key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            # value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            # out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)
            query_bias = np2th(weights[(ROOT + ATTENTION_Q + "/bias")]).view(-1)
            key_bias = np2th(weights[(ROOT+ATTENTION_K+ "/bias")]).view(-1)
            value_bias = np2th(weights[(ROOT+ ATTENTION_V+ "/bias")]).view(-1)
            out_bias = np2th(weights[(ROOT+ ATTENTION_OUT+ "/bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[(ROOT + FC_0+ "/kernel")]).t()
            mlp_weight_1 = np2th(weights[(ROOT+ FC_1+"/kernel")]).t()
            mlp_bias_0 = np2th(weights[(ROOT+FC_0+ "/bias")]).t()
            mlp_bias_1 = np2th(weights[(ROOT+ FC_1+ "/bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[(ROOT+ ATTENTION_NORM+ "/scale")]))
            self.attention_norm.bias.copy_(np2th(weights[(ROOT+ ATTENTION_NORM+ "/bias")]))

            self.attention_norm_y.weight.copy_(np2th(weights[(ROOT+ ATTENTION_NORM+ "/scale")]))
            self.attention_norm_y.bias.copy_(np2th(weights[(ROOT+ ATTENTION_NORM+ "/bias")]))

            self.ffn_norm.weight.copy_(np2th(weights[(ROOT+ MLP_NORM+ "/scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[(ROOT+MLP_NORM+"/bias")]))


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3, input_dim=128,old = 1):
        super(Embeddings, self).__init__()
        self.config = config
        img_size = _pair(img_size)
        grid_size = config.patches["grid"]
        patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)

        patch_size = config.patch_size
        patch_size_real = (config.patch_size, config.patch_size)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
        in_channels = 256
        #Learnable patch embeddings
        self.patch_embeddings = Conv2d(in_channels=input_dim,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        #learnable positional encodings
        self.positional_encoding = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        x = self.patch_embeddings(x) # x 4 128 64 64->4 1024 16 16
        x = x.flatten(2) # 4 1024 256
        x = x.transpose(-1, -2) # 4 256 1024
        embeddings = x + self.positional_encoding
        embeddings = self.dropout(embeddings)
        return embeddings