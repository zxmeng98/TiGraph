import torch
from .RevGNN.torch_vertex import GENConv
from .RevGNN.rev_layer import norm_layer
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging


class DeeperGCN(torch.nn.Module):
    def __init__(self, args):
        super(DeeperGCN, self).__init__()

        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.block = args.block

        self.checkpoint_grad = False

        hidden_channels = args.hidden_channels
        num_tasks = args.num_tasks
        conv = args.conv
        aggr = args.gcn_aggr

        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        y = args.y
        self.learn_y = args.learn_y

        self.msg_norm = args.msg_norm
        learn_msg_scale = args.learn_msg_scale

        conv_encode_edge = args.conv_encode_edge
        norm = args.norm
        mlp_layers = args.mlp_layers

        self.use_one_hot_encoding = args.use_one_hot_encoding

        # Use in pp: monitor
        self.first_stage = True
        self.last_stage = False

        # save gpu mem using gradient ckpt
        # if aggr not in ['add', 'max', 'mean'] and self.num_layers > 15:
        #     self.checkpoint_grad = True
        #     self.ckp_k = 9

        # print('The number of layers {}'.format(self.num_layers),
        #       'Aggregation method {}'.format(aggr),
        #       'block: {}'.format(self.block))
        # print(self.checkpoint_grad)

        # if self.block == 'res+':
        #     print('LN/BN->ReLU->GraphConv->Res')
        # elif self.block == 'res':
        #     print('GraphConv->LN/BN->ReLU->Res')
        # elif self.block == 'dense':
        #     raise NotImplementedError('To be implemented')
        # elif self.block == "plain":
        #     print('GraphConv->LN/BN->ReLU')
        # else:
        #     raise Exception('Unknown block Type')

        self.gcns = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()

        for layer_id in range(self.num_layers):

            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              y=y, learn_y=self.learn_y,
                              msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                              encode_edge=conv_encode_edge, edge_feat_dim=hidden_channels,
                              norm=norm, mlp_layers=mlp_layers)
            else:
                raise Exception('Unknown Conv Type')

            self.gcns.append(gcn)
            self.layer_norms.append(norm_layer(norm, hidden_channels))

        if self.use_one_hot_encoding:
            self.node_one_hot_encoder = torch.nn.Linear(args.in_size, 8)
            self.node_features_encoder = torch.nn.Linear(args.in_size+8, hidden_channels)
        else:
            self.node_features_encoder = torch.nn.Linear(args.in_size, hidden_channels)

        self.edge_encoder = torch.nn.Linear(8, hidden_channels)
        self.node_pred_linear = torch.nn.Linear(hidden_channels, args.out_size)

    def forward(self, g, x):
        u, v = g.edges()
        edge_index = torch.stack((u, v), dim=0).to(torch.int64)

        node_features_1st = x

        if self.node_features_encoder:
            if self.use_one_hot_encoding:
                node_features_2nd = self.node_one_hot_encoder(x)
                # concatenate
                node_features = torch.cat((node_features_1st, node_features_2nd), dim=1)
            else:
                node_features = node_features_1st

            h = self.node_features_encoder(node_features)
        else:
            h = node_features_1st

        if self.block == 'res+':
            if self.checkpoint_grad:
                h = self.gcns[0](h, edge_index)
                for layer in range(1, self.num_layers):
                    h1 = self.layer_norms[layer-1](h)
                    h2 = F.relu(h1)
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)
                    if layer % self.ckp_k != 0:
                        res = checkpoint(self.gcns[layer], h2, edge_index)
                        h = res + h
                    else:
                        h = self.gcns[layer](h2, edge_index) + h

            else:
                if self.first_stage:
                    h = self.gcns[0](h, edge_index) 
                    for layer in range(1, len(self.gcns)):
                        h1 = self.layer_norms[layer-1](h)
                        h2 = F.relu(h1)
                        h2 = F.dropout(h2, p=self.dropout, training=self.training)
                        h = self.gcns[layer](h2, edge_index) + h
                elif self.last_stage:
                    for layer in range(len(self.gcns)):
                        h1 = self.layer_norms[layer](h)
                        h2 = F.relu(h1)
                        h2 = F.dropout(h2, p=self.dropout, training=self.training)
                        h = self.gcns[layer](h2, edge_index) + h
                    h = F.relu(self.layer_norms[len(self.gcns)](h))
                    h = F.dropout(h, p=self.dropout, training=self.training)
                else:
                    for layer in range(len(self.gcns)):
                        h1 = self.layer_norms[layer](h)
                        h2 = F.relu(h1)
                        h2 = F.dropout(h2, p=self.dropout, training=self.training)
                        h = self.gcns[layer](h2, edge_index) + h

                h = self.node_pred_linear(h) if self.node_pred_linear else h

            return h

        elif self.block == 'res':
            # block = 'res'时，layer = 112 不收敛, 少层数才行
            if self.first_stage:
                h = F.relu(self.layer_norms[0](self.gcns[0](h, edge_index)))
                h = F.dropout(h, p=self.dropout, training=self.training)

                for layer in range(1, len(self.gcns)):
                    h1 = self.gcns[layer](h, edge_index)
                    h2 = self.layer_norms[layer-1](h1)
                    h = F.relu(h2) + h
                    h = F.dropout(h, p=self.dropout, training=self.training)
            else:
                for layer in range(len(self.gcns)):
                    h1 = self.gcns[layer](h, edge_index)
                    h2 = self.layer_norms[layer](h1)
                    h = F.relu(h2) + h
                    h = F.dropout(h, p=self.dropout, training=self.training)
            
            h = self.node_pred_linear(h) if self.node_pred_linear else h

            return h   

        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')

        elif self.block == 'plain':

            h = F.relu(self.layer_norms[0](self.gcns[0](h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index)
                h2 = self.layer_norms[layer](h1)
                h = F.relu(h2)
                h = F.dropout(h, p=self.dropout, training=self.training)

            return self.node_pred_linear(h)

        else:
            raise Exception('Unknown block Type')

    def print_params(self, epoch=None, final=False):

        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print('Final t {}'.format(ts))
            else:
                logging.info('Epoch {}, t {}'.format(epoch, ts))

        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print('Final p {}'.format(ps))
            else:
                logging.info('Epoch {}, p {}'.format(epoch, ps))

        if self.learn_y:
            ys = []
            for gcn in self.gcns:
                ys.append(gcn.sigmoid_y.item())
            if final:
                print('Final sigmoid(y) {}'.format(ys))
            else:
                logging.info('Epoch {}, sigmoid(y) {}'.format(epoch, ys))

        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print('Final s {}'.format(ss))
            else:
                logging.info('Epoch {}, s {}'.format(epoch, ss))