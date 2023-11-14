import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
from dgl.nn import GraphConv
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):  # nn.Module - родительский класс PyTorch для регистрации внутри бэка всех слоев, параметров и тд
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)  # слой свертки графа, в котором мы указываем размерность входящих и исходящих элементов. В нашем случае это вектор из слов
        self.conv2 = GraphConv(h_feats, num_classes)  # последний слой, на выходе получаем размерность = 7 - количеству классов в графе

    def forward(self, g, in_feat):  # основная функция вычислений
        h = self.conv1(g, in_feat)  # первый слой
        h = F.relu(h)  # функция ктивации
        h = self.conv2(g, h)  # выходной слой
        return h


# Create the model with given dimensions
# 16 - количество параметров на среднем слое
# model = GCN(g.ndata["feat"].shape[1], 16, dataset.num_classes)