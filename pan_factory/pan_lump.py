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


class PANLump(MessagePassing):
    def __init__(self, emb_dim):
        super(PANLump, self).__init__(aggr = "add")
        # self.mlp = torch.nn.Sequential(torch.nn.Linear(2*emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.atom_encoder = AtomEncoder(emb_dim = emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)
    def forward(self, x, edge_index, edge_attr):
        x = self.atom_encoder(x)
        edge_embedding = self.bond_encoder(edge_attr)
        # out = self.mlp(torch.cat((x, self.propagate(edge_index, x=x, edge_attr=edge_embedding)), dim=-1))
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out
    def message(self, x_j, edge_attr):
        return edge_attr
    def update(self, aggr_out):
        return aggr_out
