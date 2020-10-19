#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import torch
import torch.nn.functional as F
from scipy import linalg
import numpy as np
from torch import nn
import torch.nn.functional as F


def average_agg(w, dp):
    """
    Federated averaging
    :param w: list of client model parameters
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k] + w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w)) + torch.mul(torch.randn(w_avg[k].shape), dp)
    return w_avg


def weighted_agg(w_clients, w_server, stepsize, metric, dp):
    """
    Attentive aggregation
    :param w_clients: list of client model parameters
    :param w_server: server model parameters
    :param stepsize: step size for aggregation
    :param metric: similarity
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    w_next = copy.deepcopy(w_server)
    att, att_mat = {}, {}
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            att[k][i] = torch.from_numpy(np.array(linalg.norm(w_server[k]-w_clients[i][k], ord=metric)))
    for k in w_next.keys():
        att[k] = F.softmax(att[k], dim=0)
    for k in w_next.keys():
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            att_weight += torch.mul(w_server[k]-w_clients[i][k], att[k][i])
        w_next[k] = w_server[k] - torch.mul(att_weight, stepsize) + torch.mul(torch.randn(w_server[k].shape), dp)
    return w_next


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.args = config
        self.attn_dim = int(self.args.frac * self.args.nusers)
        self.attn1 = nn.Linear(self.attn_dim, self.attn_dim)
        self.attn2 = nn.Linear(self.attn_dim, self.attn_dim)

    def _get_atts(self, p):
        attention = torch.tanh(self.attn1(p))
        attention = torch.nn.functional.softmax(self.attn2(attention), dim=0)

        return attention

    def _weighted_sum(self, param, a):
        result = 0

        for i in range(len(param)):
            result += param[i] * a[i]

        return result

    def forward(self, w):
        w_att = copy.deepcopy(w[0])
        concat_p = torch.zeros(len(w))
        for k in w_att.keys():
            for i in range(len(w)):
                concat_p[i] = torch.mean(w[i][k] * 1000000)
            attn_score = self._get_atts(concat_p)
            w_att[k] = self._weighted_sum([w[j][k] for j in range(len(w))], attn_score)

        return w_att
