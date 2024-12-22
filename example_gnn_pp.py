import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import os
from dgl.nn import GraphConv
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe
import copy
import random
import numpy as np
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, subgraph


# 定义简单的 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleDict()
        self.layers["1"] = GraphConv(in_feats, h_feats, activation=F.relu)
        self.layers["2"] = GraphConv(h_feats, out_feats)
        self.dropout = nn.Dropout(0.5)
        # self.conv1 = GraphConv(in_feats, h_feats)
        # self.conv2 = GraphConv(h_feats, out_feats)

    def forward(self, g, h):
        # NOTE: forward里面不能一层一层hard-code conv1, conv2，才能在split的时候通过del layer删除，不然构造PipeStage会报错forward里面的某一层找不到。虽然print出来的submodule删掉了别的层，但是forward还保留了。
        # for layer in self.layers.values():
        #     h = layer(self.g, h)    
        
        ## 2. None的写法
        h = self.layers["1"](g, h) if self.layers["1"] else h
        h = self.dropout(h) if self.dropout else h
        h = self.layers["2"](g, h) if self.layers["2"] else h

        # h = self.conv1(self.g, inputs)
        # h = torch.relu(h)
        # h = self.conv2(self.g, h)
        return h


global rank, device, pp_group, stage_index, num_stages
def init_distributed():
   global rank, device, pp_group, stage_index, num_stages
   rank = int(os.environ["LOCAL_RANK"])
   world_size = int(os.environ["WORLD_SIZE"])
   device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
   dist.init_process_group()

   # This group can be a sub-group in the N-D parallel case
   pp_group = dist.new_group()
   stage_index = rank
   num_stages = world_size


def manual_model_split(model, example_input_microbatch) -> PipelineStage:
    if stage_index == 0:
        # prepare the first stage model
        # del model.layers["2"]
        model.dropout = None
        model.layers["2"] = None
        stage_input_microbatch = example_input_microbatch

    elif stage_index == 1:
        # prepare the second stage model
        # del model.layers["1"]
        model.layers["1"] = None
        # NOTE: 不管哪个stage，self.inputs的格式都要按照最开始进入forward的args来，但是h是从中间状态开始。所以这里stage 1不能只传入h，要和forward格式一样传入(g, h)
        feature_input_microbatch = torch.randn(example_input_microbatch[1].shape[0], 16)
        stage_input_microbatch = (example_input_microbatch[0], feature_input_microbatch)
        
    # print(f"{rank}, {model}")
    # exit(0)

    stage = PipelineStage(
      model,
      stage_index,
      num_stages,
      device,
      input_args=stage_input_microbatch,
   )
    return stage


def evaluate(g, features, labels, mask, model):
    model.eval() # NOTE：没有这个出来的loss会不一样
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

if __name__ == "__main__":
    init_distributed()
    num_microbatches = 1

    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # load and preprocess dataset
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    data = PubmedGraphDataset(transform=transform)
    g = data[0]
    g = g.int()
    g.remove_nodes(np.arange(5)) 
    inputs = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    all_idx = torch.arange(inputs.shape[0])
    microbatch_idxes = torch.tensor_split(all_idx, num_microbatches)
    microbatch_g = dgl.node_subgraph(g, microbatch_idxes[0].to(torch.int32)) # only extract edges between nodes in microbatch_idxes[0]
    microbatch_g.ndata.clear()
    microbatch_g.edata.clear()

    model = GCN(inputs.shape[1], 16, data.num_classes)
        # for name, param in model_copy.named_parameters():
        #     print(name, param)

    # if rank == 0:
    example_input_microbatch = (microbatch_g, inputs[microbatch_idxes[0]])
    stage = manual_model_split(model, example_input_microbatch)
    
    # pipe_gnn = tracer_model_split(model, data)

    loss_fcn = nn.CrossEntropyLoss()
    schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches, loss_fn=nn.CrossEntropyLoss())
    # exit()

    optimizer = torch.optim.Adam(stage.submod.parameters(), lr=1e-2, weight_decay=5e-4)

    g = g.to(device)
    inputs = inputs.to(device)
    labels = labels.to(device)

    # if rank == 0:
    #     schedule.step(inputs)
    # elif rank == 1:
    #     losses = []
    #     output = schedule.step(target=labels, losses=losses)

    # training loop
    loss_list, val_acc_list, loss_org_list, val_acc_org_list = [], [], [], []
    for epoch in range(200):
        optimizer.zero_grad()
        stage.submod.train()
        if rank == 0:
            # step里面要改，此处micro_batch includes: 1. features 2. g
            schedule.step(g, inputs)
        elif rank == 1:
            losses = [] # size: [num_microbatches]
            output = schedule.step(g, target=labels, losses=losses) # TODO：此时是用所有dataset训的，真实的应该只用inputs[train_mask]
            for i in range(len(losses)):
                losses[i] = losses[i].item()

            train_mask = masks[0]
            loss = loss_fcn(output, labels)
            loss_list.append(loss.item())
            # for name, param in stage.submod.named_parameters():
            #     if name == 'layers.2.weight':
            #         print(f'Epoch {epoch} {name}: {param.grad}')
            
        optimizer.step()

        stage.submod.eval()
        if rank == 0:
            schedule.step(g, inputs)
        elif rank == 1:
            losses_val = []
            output = schedule.step(g, target=labels, losses=losses_val)   
            val_output = output[masks[1]]
            val_labels = labels[masks[1]]
            _, indices = torch.max(val_output, dim=1)
            correct = torch.sum(indices == val_labels)
            acc = correct.item() * 1.0 / len(val_labels)
            print(
                "PP Epoch {:05d} | Loss {:.4f} | Val Accuracy {:.4f} ".format(
                    epoch, loss.item(), acc
                )
            )
            val_acc_list.append(acc)
            

    # test acc
    stage.submod.eval()
    if rank == 0:
        schedule.step(g, inputs)
    elif rank == 1:
        losses = []
        output = schedule.step(g, target=labels, losses=losses)
        test_output = output[masks[2]]
        test_labels = labels[masks[2]]
        _, indices = torch.max(test_output, dim=1)
        correct = torch.sum(indices == test_labels)
        acc = correct.item() * 1.0 / len(test_labels)
        print("Test accuracy {:.4f}".format(acc))
        

    dataset = 'pubmed'
    if not os.path.exists(f'./exps/{dataset}'): 
        os.makedirs(f'./exps/{dataset}')
    if rank == 1:
        np.save('./exps/' + dataset + '/gcn_pp_mb64_loss', np.array(loss_list))
        np.save('./exps/' + dataset + '/gcn_pp_mb64_val_acc', np.array(val_acc_list))
        
    # if rank == num_stages - 1:
    #     # Run the original code and get the output for comparison
    #     model_copy = model_copy.to(device)
    #     torch.manual_seed(seed)
    #     reference_output = model_copy(inputs)
    #     # Compare numerics of pipeline and original model
    #     torch.testing.assert_close(output, reference_output)
    #     print(f"Loss of microbatches: {losses}")
    #     print(" Pipeline parallel model ran successfully! ".center(80, "*"))
    #     print(output)
    #     print(reference_output)

    dist.destroy_process_group()