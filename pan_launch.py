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

from pan_factory.pan import PAN
from utils import train, eval


def main():
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
    train_acc = np.zeros(runs,dtype=np.float)
    eval_acc = np.zeros(runs,dtype=np.float)
    max_score = 0.


    for run in range(runs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset = PygGraphPropPredDataset(name=datasetname, root='dataset/')
        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)
        evaluator = Evaluator(name=datasetname)

        num_graph = len(dataset)

        ## train model
        t1 = datetime.datetime.fromtimestamp(time.time())
        model = PAN(dataset.num_node_features, dataset.num_classes, nhid=nhid, ratio=pool_ratio, filter_size=filter_size, pos_weight=pos_weight).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
        model.load_state_dict(torch.load('latest.pth'), strict=False)
        train_acc[run] = eval(model, train_loader, device, evaluator)
        eval_acc[run] = eval(model, val_loader, device, evaluator)
        test_acc[run] = eval(model, test_loader, device, evaluator)
        print('==Train Acc: {:.4f}'.format(train_acc[run]))
        print('==Test Acc: {:.4f}'.format(test_acc[run]))

        sv_prefix = datasetname + '_time' + str(t1)
        sv_results = sv_prefix + '.mat'
        sv_model = sv_prefix + '.pth'
        mdict = options.__dict__
        mdict.update({'test_acc':test_acc,
                      'val_acc':val_acc,
                      'eval_acc':eval_acc,
                      'train_loss':train_loss})
        sio.savemat(sv_results,mdict=mdict)
        torch.save(model.state_dict(), sv_model)
    print('==Mean Test Acc: {:.4f}'.format(np.mean(test_acc)))


if __name__ == '__main__':
    main()

