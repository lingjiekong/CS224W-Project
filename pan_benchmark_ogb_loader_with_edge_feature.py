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

### define convolution

class PANConv(MessagePassing):
    def __init__(self, in_channels, out_channels, filter_size=4, panconv_filter_weight=None):
        super(PANConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.filter_size = filter_size
        if panconv_filter_weight is None:
            self.panconv_filter_weight = torch.nn.Parameter(0.5 * torch.ones(filter_size), requires_grad=True)

    def forward(self, x, edge_index, edge_emb=None, num_nodes=None, edge_mask_list=None):
        # x has shape [N, in_channels]
        if edge_mask_list is None:
            AFTERDROP = False
        else:
            AFTERDROP = True

        # edge_index has shape [2, E]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        # Step 1: Path integral
        edge_index, edge_weight = self.panentropy_sparse(edge_index, edge_emb, num_nodes, AFTERDROP, edge_mask_list)

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        x_size0 = x.size(0)

        # Step 3: Compute normalization
        row, col = edge_index
        deg = degree(row, x_size0, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        norm = norm.mul(edge_weight)

        # Step 4-6: Start propagating messages.
        return self.propagate(edge_index, size=(x_size0, x_size0), x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

    def panentropy_sparse(self, edge_index, edge_emb, num_nodes, AFTERDROP, edge_mask_list):

        if edge_emb is None:
            edge_value = torch.ones(edge_index.size(1), device=edge_index.device)
        else:
            edge_value = edge_emb
        edge_index, edge_value = coalesce(edge_index, edge_value, num_nodes, num_nodes)

        # iteratively add weighted matrix power
        pan_index, pan_value = eye(num_nodes, device=edge_index.device)
        indextmp = pan_index.clone().to(edge_index.device)
        valuetmp = pan_value.clone().to(edge_index.device)

        pan_value = self.panconv_filter_weight[0] * pan_value

        for i in range(self.filter_size - 1):
            if AFTERDROP:
                indextmp, valuetmp = spspmm(indextmp, valuetmp, edge_index, edge_value * edge_mask_list[i], num_nodes, num_nodes, num_nodes)
            else:
                indextmp, valuetmp = spspmm(indextmp, valuetmp, edge_index, edge_value, num_nodes, num_nodes, num_nodes)
            valuetmp = valuetmp * self.panconv_filter_weight[i+1]
            indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            pan_index = torch.cat((pan_index, indextmp), 1)
            pan_value = torch.cat((pan_value, valuetmp))

        return coalesce(pan_index, pan_value, num_nodes, num_nodes, op='add')


### define pooling

class PANPooling(torch.nn.Module):
    r""" General Graph pooling layer based on PAN, which can work with all layers.
    """
    def __init__(self, in_channels, ratio=0.5, pan_pool_weight=None, min_score=None, multiplier=1,
                 nonlinearity=torch.tanh, filter_size=3, panpool_filter_weight=None):
        super(PANPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.filter_size = filter_size
        if panpool_filter_weight is None:
            self.panpool_filter_weight = torch.nn.Parameter(0.5 * torch.ones(filter_size), requires_grad=True)

        self.transform = Parameter(torch.ones(in_channels), requires_grad=True)

        if pan_pool_weight is None:
            #self.weight = torch.tensor([0.7, 0.3], device=self.transform.device)
            self.pan_pool_weight = torch.nn.Parameter(0.5 * torch.ones(2), requires_grad=True)
        else:
            self.pan_pool_weight = pan_pool_weight

    def forward(self, x, edge_index, batch=None, num_nodes=None):

        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # Path integral
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        edge_index, edge_weight = self.panentropy_sparse(edge_index, num_nodes)

        # weighted degree
        num_nodes = x.size(0)
        degree = torch.zeros(num_nodes, device=edge_index.device)
        degree = scatter_add(edge_weight, edge_index[0], out=degree)

        # linear transform
        xtransform = torch.matmul(x, self.transform)

        # aggregate score
        x_transform_norm = xtransform #/ xtransform.norm(p=2, dim=-1)
        degree_norm = degree #/ degree.norm(p=2, dim=-1)
        score = self.pan_pool_weight[0] * x_transform_norm + self.pan_pool_weight[1] * degree_norm

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = self.topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_weight = self.filter_adj(edge_index, edge_weight, perm, num_nodes=score.size(0))

        return x, edge_index, edge_weight, batch, perm, score[perm]

    def topk(self, x, ratio, batch, min_score=None, tol=1e-7):

        if min_score is not None:
            # Make sure that we do not drop all nodes in a graph.
            scores_max = scatter_max(x, batch)[0][batch] - tol
            scores_min = scores_max.clamp(max=min_score)

            perm = torch.nonzero(x > scores_min).view(-1)
        else:
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

            cum_num_nodes = torch.cat(
                [num_nodes.new_zeros(1),
                 num_nodes.cumsum(dim=0)[:-1]], dim=0)

            index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
            index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

            dense_x = x.new_full((batch_size * max_num_nodes, ), -2)
            dense_x[index] = x
            dense_x = dense_x.view(batch_size, max_num_nodes)

            _, perm = dense_x.sort(dim=-1, descending=True)

            perm = perm + cum_num_nodes.view(-1, 1)
            perm = perm.view(-1)

            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
            mask = [
                torch.arange(k[i], dtype=torch.long, device=x.device) +
                i * max_num_nodes for i in range(batch_size)
            ]
            mask = torch.cat(mask, dim=0)

            perm = perm[mask]

        return perm

    def filter_adj(self, edge_index, edge_weight, perm, num_nodes=None):

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        mask = perm.new_full((num_nodes, ), -1)
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        mask[perm] = i

        row, col = edge_index
        row, col = mask[row], mask[col]
        mask = (row >= 0) & (col >= 0)
        row, col = row[mask], col[mask]

        if edge_weight is not None:
            edge_weight = edge_weight[mask]

        return torch.stack([row, col], dim=0), edge_weight

    def panentropy_sparse(self, edge_index, num_nodes):

        edge_value = torch.ones(edge_index.size(1), device=edge_index.device)
        edge_index, edge_value = coalesce(edge_index, edge_value, num_nodes, num_nodes)

        # iteratively add weighted matrix power
        pan_index, pan_value = eye(num_nodes, device=edge_index.device)
        indextmp = pan_index.clone().to(edge_index.device)
        valuetmp = pan_value.clone().to(edge_index.device)

        pan_value = self.panpool_filter_weight[0] * pan_value

        for i in range(self.filter_size - 1):
            #indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            indextmp, valuetmp = spspmm(indextmp, valuetmp, edge_index, edge_value, num_nodes, num_nodes, num_nodes)
            valuetmp = valuetmp * self.panpool_filter_weight[i+1]
            indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            pan_index = torch.cat((pan_index, indextmp), 1)
            pan_value = torch.cat((pan_value, valuetmp))

        return coalesce(pan_index, pan_value, num_nodes, num_nodes, op='add')

### define dropout

class PANDropout(torch.nn.Module):
    def __init__(self, filter_size=4):
        super(PANDropout, self).__init__()

        self.filter_size =filter_size

    def forward(self, edge_index, p=0.5):
        # p - probability of an element to be zeroed

        # sava all network
        #edge_mask_list = []
        edge_mask_list = torch.empty(0)
        edge_mask_list.to(edge_index.device)

        num = edge_index.size(1)
        bern = torch.distributions.bernoulli.Bernoulli(torch.tensor([p]))

        for i in range(self.filter_size - 1):
            edge_mask = bern.sample([num]).squeeze()
            #edge_mask_list.append(edge_mask)
            edge_mask_list = torch.cat([edge_mask_list, edge_mask])

        return True, edge_mask_list


### build model

class PAN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, nhid, ratio, filter_size, pos_weight=1):
        super(PAN, self).__init__()
        self.atom_encoder = AtomEncoder(nhid)
        self.edge_encoder = BondEncoder(nhid)
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

        x = self.atom_encoder(x)
        edge_emb = None
        x = self.conv1(x, edge_index, edge_emb=edge_emb)
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


def train(model, opt, loader, device):
    model.train()

    loss_all = 0
    for batch in loader:
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        batch.to(device)
        opt.zero_grad()
        pred,_ = model(batch)
        loss = model.loss(pred, batch.y.type(torch.float32))
        loss.backward()
        opt.step()
        loss_all += batch.num_graphs * loss.item()

        for name, param in model.named_parameters():
            # if 'pan_pool_weight' in name:
            #     param.data = param.data.clamp(0, 1)
            if 'panconv_filter_weight' in name:
                param.data = param.data.clamp(0, 1)
            if 'panpool_filter_weight' in name:
                param.data = param.data.clamp(0, 1)
    return loss_all / len(loader.dataset)


def test(model, loader, device, evaluator):
    model.eval()

    total_score = 0
    count_batch = 0
    for batch in loader:
        batch.to(device)
        pred,_ = model(batch)
        pred = torch.sigmoid(pred)
        input_dict = {"y_true": batch.y.cpu().detach().numpy(), "y_pred": pred.cpu().detach().numpy()}
        total_score += evaluator.eval(input_dict)["rocauc"]
        count_batch += 1
    
    score = total_score/count_batch

    return score

def eval(model, loader, device, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred,_ = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["rocauc"]

parser = OptionParser()
parser.add_option("--dataset_name",
                  dest="dataset_name", default='ogbg-molhiv',
                  help="the name of dataset from Pytorch Geometric, other options include PROTEINS_full, NCI1, AIDS, Mutagenicity")
parser.add_option("--runs",
                  dest="runs", default=1, type=np.int,
                  help="number of runs")
parser.add_option("--batch_size", type=np.int,
                  dest="batch_size", default=128,
                  help="batch size")
parser.add_option("--L",
                  dest="L", default=3, type=np.int,
                  help="order L in MET")
parser.add_option("--learning_rate", type=np.float,
                  dest="learning_rate", default=0.005,
                  help="learning rate")
parser.add_option("--weight_decay", type=np.float,
                  dest="weight_decay", default=1e-3,
                  help="weight decay")
parser.add_option("--pool_ratio", type=np.float,
                  dest="pool_ratio", default=0.9,
                  help="proportion of nodes to be pooled")
parser.add_option("--nhid", type=np.int,
                  dest="nhid", default=64,
                  help="number of each hidden-layer neurons")
parser.add_option("--epochs", type=np.int,
                  dest="epochs", default=20,
                  help="number of epochs each run")
parser.add_option("--pos_weight", type=np.float,
                  dest="pos_weight", default=8.,
                  help="pos_weight in BCEWithLogitsLoss")

options, argss = parser.parse_args()
datasetname = options.dataset_name
runs = options.runs
batch_size = options.batch_size
filter_size = options.L+1
learning_rate = options.learning_rate
weight_decay = options.weight_decay
pool_ratio = options.pool_ratio
nhid = options.nhid
epochs = options.epochs
pos_weight = options.pos_weight

train_loss = np.zeros((runs,epochs),dtype=np.float)
val_acc = np.zeros((runs,epochs),dtype=np.float)
test_acc = np.zeros(runs,dtype=np.float)
max_score = 0.

# dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PygGraphPropPredDataset(name=datasetname, root='dataset/')
split_idx = dataset.get_idx_split()
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)
evaluator = Evaluator(name=datasetname)

num_graph = len(dataset)

## train model
    
model = PAN(dataset.num_node_features, dataset.num_classes, nhid=nhid, ratio=pool_ratio, filter_size=filter_size, pos_weight=pos_weight).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

for run in range(runs):
    for epoch in range(epochs):
        # training
        loss = train(model, optimizer, train_loader, device)
        train_loss[run, epoch] = loss

        # train
        #train_a = eval(model, train_loader, device, evaluator)

        # validation
        val_acc_1 = eval(model, val_loader, device, evaluator)
        val_acc[run, epoch] = val_acc_1
        # print('Val Run: {:02d}, Epoch: {:03d}, Val loss: {:.4f}, Val acc: {:.4f}'.format(run+1,epoch+1,val_loss[run,epoch],val_acc[run,epoch]))

        print('Epoch: {:03d}, Train loss: {:.4f}, Val acc: {:.4f}'.format(epoch + 1, loss, val_acc_1))

        if val_acc_1 > max_score:
            # save the model and reuse later in test
            torch.save(model.state_dict(), 'latest.pth')
            max_score = val_acc_1

    # test
    model.load_state_dict(torch.load('latest.pth'))
    test_acc[run] = eval(model, test_loader, device, evaluator)
    print('==Test Acc: {:.4f}'.format(test_acc[run]))

print('==Mean Test Acc: {:.4f}'.format(np.mean(test_acc)))

t1 = time.time()
sv = datasetname + '_pcpa_runs' + str(runs) + '_time' + str(t1) + '.mat'
sio.savemat(sv,mdict={'test_acc':test_acc,'val_acc':val_acc,'train_loss':train_loss,'filter_size':filter_size,'learning_rate':learning_rate,'weight_decay':weight_decay,'nhid':nhid,'batch_size':batch_size,'epochs':epochs})







