import torch
import torch.nn.functional as F

from torch.nn import Parameter
from torch.nn import BCEWithLogitsLoss
from torch_scatter import scatter_add, scatter_max, scatter_mean

from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing
from torch_geometric.data import DataLoader, Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch.utils.data import random_split

from torch_sparse import spspmm
from torch_sparse import coalesce
from torch_sparse import eye

#from collections import OrderedDict

import os
import scipy.io as sio
import numpy as np
from optparse import OptionParser
import time
# add ogb data loader
#from ogb.graphproppred import GraphPropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from sklearn.metrics import roc_auc_score
import datetime

from pan_factory.pan_conv import PANConv
from pan_factory.pan_pooling import PANPooling
from pan_factory.pan_lump import PANLump


class PAN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, nhid, ratio, filter_size, pos_weight=1):
        super(PAN, self).__init__()
        self.panlump = PANLump(nhid)
        self.conv1 = PANConv(nhid, nhid, filter_size)
        self.pool1 = PANPooling(nhid, filter_size=filter_size)
        #self.drop1 = PANDropout()
        self.conv2 = PANConv(nhid, nhid, filter_size=2)
        self.pool2 = PANPooling(nhid)
        #self.drop2 = PANDropout()
        self.conv3 = PANConv(nhid, nhid, filter_size=2)
        self.pool3 = PANPooling(nhid)
        
        self.lin1 = torch.nn.Linear(nhid, nhid//2)
        self.lin2 = torch.nn.Linear(nhid//2, 1)

        self.mlp = torch.nn.Linear(nhid, num_classes)
        self.loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        #self.loss_fn = BCEWithLogitsLoss()

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        perm_list = list()
        edge_mask_list = None

        x = self.panlump(x, edge_index, edge_attr)
        x = self.conv1(x, edge_index)
        x, edge_index, _, batch, perm, score_perm = self.pool1(x, edge_index, batch=batch)
        perm_list.append(perm)

        #AFTERDROP, edge_mask_list = self.drop1(edge_index, p=0.5)
        x = self.conv2(x, edge_index, edge_mask_list=edge_mask_list)
        x, edge_index, _, batch, perm, score_perm = self.pool2(x, edge_index, batch=batch)
        perm_list.append(perm)

        #AFTERDROP, edge_mask_list = self.drop2(edge_index, p=0.5)
        x = self.conv3(x, edge_index, edge_mask_list=edge_mask_list)
        x, edge_index, _, batch, perm, score_perm = self.pool3(x, edge_index, batch=batch)
        perm_list.append(perm)
        
        mean = scatter_mean(x, batch, dim=0)
        x = mean
        
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        #x = self.mlp(x)
        #x = F.log_softmax(x, dim=-1)

        return x, perm_list

    def loss(self, logistic, label):
        return self.loss_fn(logistic, label)

