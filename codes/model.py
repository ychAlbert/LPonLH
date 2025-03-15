from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
from os import path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
from torch.utils import data
from torch.utils.data import DataLoader
import time



def get_param(shape):
	param = nn.Parameter(torch.Tensor(*shape)); 	
	nn.init.xavier_normal_(param.data)
	return param




class BaseLayer(nn.Module):
    def __init__(self, args, in_feat, out_feat):
        super(BaseLayer, self).__init__()

        self.lamda = args.lamda
        self.w = nn.Linear(in_feat, out_feat)
        self.w_type = nn.Linear(in_feat, args.t_embed)

        self.gamma = nn.Linear(args.t_embed, in_feat) 
        self.beta = nn.Linear(args.t_embed, in_feat)

       
        self.no_film = args.no_film


    def forward(self, embed, idx, paths, masks, neighs, t_tilde=None, ntype=None):
        
      
        t_x = self.w_type(embed) # [batch, t_embed size]
        
        # paths embedding 
        t_p = torch.sum(t_x[paths] * masks.unsqueeze(dim=-1), dim=-2)
        t_p /= torch.sum(masks, dim=2, keepdim=True)
   

        feat = embed[idx] 
        h_l = embed[neighs[:,:,0]]

        # path embed of neighbor node:
        if self.no_film:
            p_x = h_l
        else:
            gamma = F.leaky_relu(self.gamma(t_p))
            beta = F.leaky_relu(self.beta(t_p))
            p_x = (gamma + 1) * h_l + beta
        

        # message
        l_p = torch.unsqueeze(neighs[:,:,1],2) 
        alpha = torch.exp(-self.lamda*l_p)

        a_x = torch.sum(alpha * p_x, dim=1) 

        # final embed 
        update = (feat + a_x) / (neighs.shape[1]+1) 
        output = F.leaky_relu(self.w(update))

        L_film = torch.sum(torch.norm(gamma, dim=1)) + torch.sum(torch.norm(beta, dim=1))
        L_film /= gamma.shape[0]

        return output, 0, 0, [t_x[idx], L_film, 0]

class BaseModel(nn.Module):
    def __init__(self, data, args):
        super(BaseModel, self).__init__()

        self.use_emb = args.use_emb
        self.no_type = args.no_type
        self.op_rnn = args.op_rnn
        self.op_mean = args.op_mean

        self.reduce = True
        if self.use_emb:
            self.embed = get_param((data.num_ent, args.embed))           
            in_feat = args.embed
        else:
            if data.feat.shape[1] > 1000:
                self.fc = nn.Linear(data.feat.shape[1], args.embed)
                in_feat = args.embed
            else:
                self.reduce = False
                in_feat = data.feat.shape[1]
            

        self.layer1 = BaseLayer(args, in_feat, args.hidden)
        self.layer2 = BaseLayer(args, args.hidden, args.hidden)


        self.num = data.num_ent
        self.drop = args.drop

       
        if self.op_rnn:
            self.rnn = nn.RNN(args.t_embed, args.hidden, batch_first=True)
        elif self.op_mean:
            self.mlp = nn.Linear(args.t_embed, args.hidden)

        self.gamma = torch.cuda.FloatTensor([args.gamma]) 


    def forward(self, b_data, x=None):
       
        if self.use_emb:
            x = self.embed
        else:
            if self.reduce:
                x = self.fc(x)

        x1, l1_t1, l1_t2, sup1 = \
             self.layer1(x, b_data['l1'], b_data['paths_l1'],\
                        b_data['mask_l1'] ,b_data['end_l1'])
        t1, film1, e1 = sup1

        x1 = F.normalize(x1)
        x1 = F.dropout(x1, self.drop, training=self.training)

        x2, l2_t1, l2_t2, sup2 = \
            self.layer2(x1, b_data['l2'], b_data['paths_l2'], \
                        b_data['mask_l2'], b_data['end_l2']) 
        x2 = F.normalize(x2)
        t2, film2, e2 = sup2
        
        return  x2, t2, film1+film2  
    

    
    def new_loss(self, output, type, idx):

        a = output[idx[:,0]]            
        b = output[idx[:,1]]
        c = output[idx[:,2]]
        
        if self.no_type:
            pos = torch.norm(a - b, p=2, dim=1)
            neg = torch.norm(a - c, p=2, dim=1)
        else:

            if self.op_mean:
                t_a = type[idx[:,0]]
                t_b = type[idx[:,1]]
                t_c = type[idx[:,2]]

                t_a = F.leaky_relu(self.mlp(t_a))
                t_b = F.leaky_relu(self.mlp(t_b))
                t_c = F.leaky_relu(self.mlp(t_c))

                t_ab = (t_a + t_b)/2 
                t_ac = (t_a + t_c)/2 

            elif self.op_rnn: 
                t_a = torch.unsqueeze(type[idx[:,0]], dim=1)
                t_b = torch.unsqueeze(type[idx[:,1]], dim=1)
                t_c = torch.unsqueeze(type[idx[:,2]], dim=1)

                t_ab = torch.cat((t_a, t_b), dim=1)
                t_ac = torch.cat((t_a, t_c), dim=1)
                
                t_ab, _ = self.rnn(t_ab)
                t_ac, _ = self.rnn(t_ac)

                t_ab = t_ab[:,-1,:]
                t_ac = t_ac[:,-1,:]

            pos = torch.norm(a + t_ab - b, p=2, dim=1)
            neg = torch.norm(a + t_ac - c, p=2, dim=1)

        loss = torch.max((pos - neg), -self.gamma).mean() + self.gamma
        return loss 

class AdvancedLayer(nn.Module):
    def __init__(self, args, in_feat, out_feat):
        super(AdvancedLayer, self).__init__()

        self.lamda = args.lamda
        self.w = nn.Linear(in_feat, out_feat)
        self.w_type = nn.Linear(in_feat, args.t_embed)

        # FiLM参数
        self.no_film = args.no_film
        if not self.no_film:
            self.gamma = nn.Linear(args.t_embed, in_feat)
            self.beta = nn.Linear(args.t_embed, in_feat)

        # 消融实验参数
        self.no_gat = args.no_gat  # 禁用图注意力网络
        self.no_gate = args.no_gate  # 禁用门控机制
        self.no_path_attn = args.no_path_attn  # 禁用路径注意力
        
        # 消融实验模式
        if args.ablation_mode != 'none':
            self.no_gat = args.ablation_mode != 'gat_only'
            self.no_gate = args.ablation_mode != 'gate_only'
            self.no_path_attn = args.ablation_mode != 'path_attn_only'

        # 图注意力网络 (GAT)
        if not self.no_gat:
            self.attn_heads = args.attn_heads
            self.attn_vec = nn.Linear(in_feat, in_feat // self.attn_heads)
            self.attn_combine = nn.Linear(in_feat, in_feat)

        # 门控机制
        if not self.no_gate:
            self.gate = nn.Linear(in_feat * 2, 1)

        # 路径注意力
        if not self.no_path_attn:
            self.path_attn = nn.Linear(args.t_embed, 1)

    def forward(self, embed, idx, paths, masks, neighs, t_tilde=None, ntype=None):
        t_x = self.w_type(embed)

        # 路径注意力加权路径嵌入
        if not self.no_path_attn:
            path_attn_weights = torch.softmax(self.path_attn(t_x[paths]), dim=-2)
            t_p = torch.sum(t_x[paths] * path_attn_weights * masks.unsqueeze(dim=-1), dim=-2)
        else:
            # 不使用路径注意力，直接平均
            t_p = torch.sum(t_x[paths] * masks.unsqueeze(dim=-1), dim=-2)
        
        t_p /= torch.sum(masks, dim=2, keepdim=True) + 1e-10

        feat = embed[idx]
        h_l = embed[neighs[:, :, 0]]

        # FiLM层
        if self.no_film:
            p_x = h_l
            L_film = torch.tensor(0.0, device=feat.device)
        else:
            gamma = F.leaky_relu(self.gamma(t_p))
            beta = F.leaky_relu(self.beta(t_p))
            p_x = (gamma + 1) * h_l + beta
            L_film = torch.sum(torch.norm(gamma, dim=1)) + torch.sum(torch.norm(beta, dim=1))
            L_film /= gamma.shape[0]

        # 图注意力网络
        if not self.no_gat and h_l.size(1) > 1:
            attn_q = self.attn_vec(feat).unsqueeze(1).repeat(1, h_l.size(1), 1)
            attn_k = self.attn_vec(p_x)
            attn_v = p_x

            attn_scores = torch.sum(attn_q * attn_k, dim=-1)
            attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(2)

            attn_output = torch.sum(attn_weights * attn_v, dim=1)
            attn_output = self.attn_combine(attn_output)
        else:
            attn_output = torch.zeros_like(feat)

        # 基于路径长度的消息传递
        l_p = torch.unsqueeze(neighs[:, :, 1], 2)
        alpha = torch.exp(-self.lamda * l_p)

        gcn_x = p_x
        a_x = torch.sum(alpha * gcn_x, dim=1)

        # 门控机制
        if not self.no_gate:
            combined = torch.cat([a_x, attn_output], dim=1)
            gate_val = torch.sigmoid(self.gate(combined))
            fused = gate_val * a_x + (1 - gate_val) * attn_output
        else:
            # 不使用门控，直接相加
            fused = a_x + attn_output

        update = feat + fused
        output = F.leaky_relu(self.w(update))

        return output, 0, 0, [t_x[idx], L_film, 0]

class AdvancedModel(BaseModel):
    def __init__(self, data, args):
        super(AdvancedModel, self).__init__(data, args)

        if self.use_emb:
            in_feat = args.embed
        else:
            if data.feat.shape[1] > 1000:
                in_feat = args.embed
            else:
                in_feat = data.feat.shape[1]

        self.layer1 = AdvancedLayer(args, in_feat, args.hidden)
        self.layer2 = AdvancedLayer(args, args.hidden, args.hidden)

        # 消融实验参数
        self.no_global_attn = args.no_global_attn
        self.no_layer_norm = args.no_layer_norm
        
        # 消融实验模式
        if args.ablation_mode != 'none':
            self.no_global_attn = args.ablation_mode != 'global_attn_only'
            self.no_layer_norm = args.ablation_mode != 'layer_norm_only'

        # 全局注意力
        if not self.no_global_attn:
            self.global_attention = nn.MultiheadAttention(args.hidden, args.attn_heads, dropout=args.drop)

        # 层归一化
        if not self.no_layer_norm:
            self.layer_norm1 = nn.LayerNorm(args.hidden)
            self.layer_norm2 = nn.LayerNorm(args.hidden)

    def forward(self, b_data, x=None):
        if self.use_emb:
            x = self.embed
        else:
            if self.reduce:
                x = self.fc(x)

        x1, l1_t1, l1_t2, sup1 = \
            self.layer1(x, b_data['l1'], b_data['paths_l1'], \
                        b_data['mask_l1'], b_data['end_l1'])
        t1, film1, e1 = sup1

        # 层归一化
        if not self.no_layer_norm:
            x1 = self.layer_norm1(x1)
        
        x1 = F.dropout(x1, self.drop, training=self.training)

        x2, l2_t1, l2_t2, sup2 = \
            self.layer2(x1, b_data['l2'], b_data['paths_l2'], \
                        b_data['mask_l2'], b_data['end_l2'])

        # 层归一化
        if not self.no_layer_norm:
            x2 = self.layer_norm2(x2)
            
        t2, film2, e2 = sup2

        # 全局注意力
        if not self.no_global_attn and len(b_data['l2']) > 1:
            x2_reshaped = x2.unsqueeze(1)
            attn_output, _ = self.global_attention(x2_reshaped, x2_reshaped, x2_reshaped)
            x2 = x2 + attn_output.squeeze(1)
        
        x2 = F.normalize(x2)

        return x2, t2, film1 + film2