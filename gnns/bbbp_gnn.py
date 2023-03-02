import sys
import time
import random
import argparse
import os
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import ModuleList
from torch.nn import Sequential as Seq, ReLU, Linear as Lin,Softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINEConv, BatchNorm, global_mean_pool, GINConv, GCNConv,GATConv
from utils import set_seed, Gtrain, Gtest
sys.path.append('..')
from gnns.overloader import overload
from datasets.mutag_dataset import Mutagenicity
from utils import set_seed
from datasets import bbbp
import torch.nn.functional as F
def _Gtrain(train_loader,
           model,
           optimizer,
           device,
           criterion
           ):
    model.train()
    loss_all = 0
    criterion = criterion

    for data in train_loader:
        data.to(device)
        optimizer.zero_grad()
        out = model(data.x,
                    data.edge_index,
                    data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_loader.dataset)


def _Gtest(test_loader,
          model,
          device,
          criterion
          ):

    model.eval()
    error = 0
    correct = 0

    with torch.no_grad():

        for data in test_loader:
            data = data.to(device)
            output = model(data.x,
                           data.edge_index,
                           data.batch,
                           )

            error += criterion(output, data.y) * data.num_graphs
            correct += float(output.argmax(dim=1).eq(data.y).sum().item())

        return error / len(test_loader.dataset), correct / len(test_loader.dataset)

def parse_args():
    parser = argparse.ArgumentParser(description="Train bbbp Model")

    parser.add_argument('--model_path', nargs='?', default=osp.join(osp.dirname(__file__), 'param', 'gnns'),
                        help='path for saving trained model.')
    parser.add_argument('--cuda', type=int, default=6,
                        help='GPU device.')
    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of epoch.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Interval of evaluation.')
    parser.add_argument('--num_unit', type=int, default=4,
                        help='number of Convolution layers(units)')
    parser.add_argument('--random_label', type=bool, default=False,
                        help='train a model under label randomization for sanity check')
    parser.add_argument('--with_attr', type=bool, default=False,
                        help='train a model with edge attributes')

    return parser.parse_args()


class BBBP_GCN(torch.nn.Module):
    def __init__(self, conv_unit=3):
        super(BBBP_GCN, self).__init__()
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()
        self.edge_emb = Lin(3, 1)
        self.convs.append(GCNConv(in_channels=9, out_channels=128))
        for i in range(conv_unit - 2):
            self.convs.append(GCNConv(in_channels=128, out_channels=128))
        self.convs.append(GCNConv(in_channels=128, out_channels=128))

        self.batch_norms.extend([BatchNorm(128)] * conv_unit)
        self.relus.extend([ReLU()] * conv_unit)
        self.edge_emb = Lin(3, 128)
        # self.lin1 = Lin(128, 128)
        self.ffn = nn.Sequential(*(
                [nn.Linear(128, 128)] +
                [ReLU(), nn.Dropout(), nn.Linear(128, 2)]
        ))

        self.softmax = Softmax(dim=1)

    def forward(self, x, edge_index, batch):
        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
        # edge_weight = self.edge_emb(edge_attr).squeeze(-1)
        for conv, batch_norm, ReLU in \
                zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_weight=edge_weight)
            # x = ReLU(batch_norm(x))
            x = ReLU(x)
        graph_x = global_mean_pool(x, batch)
        pred = self.ffn(graph_x)
        self.readout = self.softmax(pred)
        return pred

    def get_node_reps(self, x, edge_index):
        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
        for conv, batch_norm, ReLU in \
                zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_weight)
            # x = ReLU(batch_norm(x))
            x = ReLU(x)
        node_x = x
        return node_x

    def get_graph_rep(self, x, edge_index, batch):
        node_x = self.get_node_reps(x, edge_index)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_pred(self, x, edge_index, batch):
        graph_x = self.get_graph_rep(x, edge_index, batch)
        pred = self.ffn(graph_x)
        self.readout = self.softmax(pred)
        return self.readout, pred

    def get_emb(self, x, edge_index, batch):
        graph_x = self.get_graph_rep(x, edge_index, batch)
        pred = self.ffn[0](graph_x)
        pred = F.relu(pred)
        return pred

    def get_pred_explain(self, x, edge_index, edge_mask, batch):
        edge_mask = edge_mask.sigmoid()
        # edge_weight = edge_mask.unsqueeze(-1).repeat(1, 128)
        for conv, batch_norm, ReLU in \
                zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_weight=edge_mask)
            x = ReLU(x)
        node_x = x
        graph_x = global_mean_pool(node_x, batch)
        pred = self.ffn(graph_x)
        self.readout = self.softmax(pred)
        return self.readout, pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)



class BBBP_GCN_attr(torch.nn.Module):
    def __init__(self, conv_unit=3):
        super(BBBP_GCN_attr, self).__init__()
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()
        self.edge_emb = nn.Linear(3, 1)
        self.convs.append(GCNConv(in_channels=9, out_channels=128))
        for i in range(conv_unit - 2):
            self.convs.append(GCNConv(in_channels=128, out_channels=128))
        self.convs.append(GCNConv(in_channels=128, out_channels=128))

        self.batch_norms.extend([BatchNorm(128)] * conv_unit)
        self.relus.extend([ReLU()] * conv_unit)
        # self.lin1 = Lin(128, 128)
        self.ffn = nn.Sequential(*(
                [nn.Linear(128, 128)] +
                [ReLU(), nn.Dropout(), nn.Linear(128, 2)]
        ))
        self.softmax = Softmax(dim=1)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device) if edge_attr == None else edge_attr
        for conv, batch_norm, ReLU in \
                zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_weight=edge_weight)
            # x = ReLU(batch_norm(x))
            x = ReLU(x)
        x = F.dropout(x, p=0.4)
        graph_x = global_mean_pool(x, batch)
        pred = self.ffn(graph_x)
        self.readout = self.softmax(pred)
        return pred

    def get_pred(self, x, edge_index, edge_attr, batch):
        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device) if edge_attr == None else edge_attr
        for conv, batch_norm, ReLU in \
                zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_weight)
            # x = ReLU(batch_norm(x))
            x = ReLU(x)
        x = F.dropout(x, p=0.4)
        node_x = x
        graph_x = global_mean_pool(node_x, batch)
        pred = self.ffn(graph_x)
        self.readout = self.softmax(pred)
        return self.readout, pred


    def get_emb(self, x, edge_index, edge_attr, batch):
        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device) if edge_attr == None else edge_attr
        for conv, batch_norm, ReLU in \
                zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_weight)
            # x = ReLU(batch_norm(x))
            x = ReLU(x)
        x = F.dropout(x, p=0.4)
        node_x = x
        graph_x = global_mean_pool(node_x, batch)
        pred = self.ffn[0](graph_x)
        pred = F.relu(pred)
        return pred

    def get_pred_explain(self, x, edge_index, edge_attr, edge_mask, batch):
        edge_mask = edge_mask.sigmoid()
        edge_mask = edge_mask * edge_attr
        # edge_weight = edge_mask.unsqueeze(-1).repeat(1, 128)
        for conv, batch_norm, ReLU in \
                zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_weight=edge_mask)
            x = ReLU(x)
        x = F.dropout(x, p=0.4)
        node_x = x
        graph_x = global_mean_pool(node_x, batch)
        pred = self.ffn(graph_x)
        self.readout = self.softmax(pred)
        return self.readout, pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)


if __name__ == '__main__':

    set_seed(0)
    args = parse_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    # folder = os.path.join("data", 'bbbp')
    folder = osp.join(osp.dirname(__file__), '..', 'data', 'bbbp')
    dataset = bbbp(folder)
    test_dataset = dataset[:200]
    val_dataset = dataset[200:400]
    train_dataset = dataset[400:]
    # train_dataset, val_dataset, test_dataset = get_datasets(name="bbbp", root="data/")

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False
                             )
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False
                            )
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True
                              )
    model = BBBP_GCN_attr(args.num_unit).to(device) if args.with_attr else BBBP_GCN(args.num_unit).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr
                                 )
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.8,
                                  patience=10,
                                  min_lr=1e-5
                                  )
    min_error = None
    for epoch in range(1, args.epoch + 1):

        t1 = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']

        loss = Gtrain(train_loader,
                      model,
                      optimizer,
                      device=device,
                      criterion=nn.CrossEntropyLoss()
                      )

        _, train_acc = Gtest(train_loader,
                             model,
                             device=device,
                             criterion=nn.CrossEntropyLoss()
                             )

        val_error, val_acc = Gtest(val_loader,
                                   model,
                                   device=device,
                                   criterion=nn.CrossEntropyLoss()
                                   )
        scheduler.step(val_error)
        if min_error is None or val_error <= min_error:
            min_error = val_error

        t2 = time.time()

        if epoch % args.verbose == 0:
            test_error, test_acc = Gtest(test_loader,
                                         model,
                                         device=device,
                                         criterion=nn.CrossEntropyLoss()
                                         )
            t3 = time.time()
            print('Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Test Loss: {:.5f}, '
                  'Test acc: {:.5f}'.format(epoch, t3 - t1, lr, loss, test_error, test_acc))
            continue

        print('Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Train acc: {:.5f}, Validation Loss: {:.5f}, '
              'Validation acc: {:5f}'.format(epoch, t2 - t1, lr, loss, train_acc, val_error, val_acc))

    if args.with_attr:
        save_path = 'bbbp_gcn_attr.pt'
    else:
        save_path = 'bbbp_gcn.pt'

    if not osp.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(model.cpu(), osp.join(args.model_path, save_path))