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


# 定义简单的 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, g):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleDict()
        self.layers["1"] = GraphConv(in_feats, h_feats, activation=F.relu)
        self.layers["2"] = GraphConv(h_feats, out_feats)
        self.dropout = nn.Dropout(0.5)
        # self.conv1 = GraphConv(in_feats, h_feats)
        # self.conv2 = GraphConv(h_feats, out_feats)
        self.g = g

    def forward(self, h):
        # NOTE: forward里面不能一层一层hard-code conv1, conv2，才能在split的时候通过del layer删除，不然构造PipeStage会报错forward里面的某一层找不到。虽然print出来的submodule删掉了别的层，但是forward还保留了。
        # for layer in self.layers.values():
        #     h = layer(self.g, h)    
        
        ## 2. None的写法
        h = self.layers["1"](self.g, h) if self.layers["1"] else h
        h = self.dropout(h) if self.dropout else h
        h = self.layers["2"](self.g, h) if self.layers["2"] else h

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
        stage_input_microbatch = torch.randn(example_input_microbatch.shape[0], 16)
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


init_distributed()

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
g = g.int().to(device)
inputs = g.ndata["feat"]
labels = g.ndata["label"]
masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]


# # 创建示例图
# g = dgl.graph(([0, 1, 1, 2], [1, 0, 2, 1]))
# g = dgl.add_self_loop(g).to(device)
# inputs = torch.randn((3, 3))
# labels = torch.tensor([0, 1, 0])

# 实例化模型

model = GCN(inputs.shape[1], 16, data.num_classes, g)
if rank == 1:
    model_copy = copy.deepcopy(model)
    model_copy = model_copy.to(device)

    # for name, param in model_copy.named_parameters():
    #     print(name, param)

# if rank == 0:
stage = manual_model_split(model, inputs)
# pipe_gnn = tracer_model_split(model, data)

loss_fcn = nn.CrossEntropyLoss()
schedule = ScheduleGPipe(stage, n_microbatches=1, loss_fn=nn.CrossEntropyLoss())
optimizer = torch.optim.Adam(stage.submod.parameters(), lr=1e-2, weight_decay=5e-4)
if rank == 1:
    optimizer_copy = torch.optim.Adam(model_copy.parameters(), lr=1e-2, weight_decay=5e-4)

inputs = inputs.to(device)
labels = labels.to(device)

# if rank == 0:
#     schedule.step(inputs)
# elif rank == 1:
#     losses = []
#     output = schedule.step(target=labels, losses=losses)

loss_list, val_acc_list, loss_org_list, val_acc_org_list = [], [], [], []
# training loop
for epoch in range(200):
    torch.manual_seed(seed)
    optimizer.zero_grad()
    stage.submod.train()
    if rank == 0:
        schedule.step(inputs)
    elif rank == 1:
        losses = []
        output = schedule.step(target=labels, losses=losses) # TODO：此时是用所有dataset训的，真实的应该只用inputs[train_mask]

        train_mask = masks[0]
        loss = loss_fcn(output, labels)
        loss_list.append(loss.item())
        # for name, param in stage.submod.named_parameters():
        #     if name == 'layers.2.weight':
        #         print(f'Epoch {epoch} {name}: {param.grad}')

        #### original model train
        optimizer_copy.zero_grad()
        torch.manual_seed(seed)
        model_copy.train()
        reference_output = model_copy(inputs)
        loss_org = loss_fcn(reference_output, labels)
        loss_org_list.append(loss_org.item())
        loss_org.backward()

        # for name, param_org in model_copy.named_parameters():
        #     if name == 'layers.2.weight':
        #         print(f'Org {name}: {param_org.grad}')

        # assert torch.allclose(param_org.grad, param.grad, atol=1e-4, rtol=1e-3)

        # print(
        #     "Epoch {:05d} | ORG Loss {:.4f} | PP Loss {:.4f}, {:.4f} ".format(
        #         epoch, loss_org.item(), losses[0].item(), loss.item()
        #     )
        # )
        
        optimizer_copy.step()
        acc = evaluate(g, inputs, labels, masks[1], model_copy)
        print(
            "ORG Epoch {:05d} | Loss {:.4f} | Val Accuracy {:.4f} ".format(
                epoch, loss_org.item(), acc
            )
        )
        val_acc_org_list.append(acc)
    optimizer.step()

    stage.submod.eval()
    if rank == 0:
        schedule.step(inputs)
    elif rank == 1:
        losses_val = []
        output = schedule.step(target=labels, losses=losses_val)   
        val_output = output[masks[1]]
        val_labels = labels[masks[1]]
        _, indices = torch.max(val_output, dim=1)
        correct = torch.sum(indices == val_labels)
        acc = correct.item() * 1.0 / len(val_labels)
        print(
            "PP Epoch {:05d} | Loss {:.4f} | Val Accuracy {:.4f} ".format(
                epoch, losses[0].item(), acc
            )
        )
        val_acc_list.append(acc)
        

# test acc
if rank == 0:
    schedule.step(inputs)
elif rank == 1:
    losses = []
    output = schedule.step(target=labels, losses=losses)
    test_output = output[masks[2]]
    test_labels = labels[masks[2]]
    _, indices = torch.max(test_output, dim=1)
    correct = torch.sum(indices == test_labels)
    acc = correct.item() * 1.0 / len(test_labels)
    print("Test accuracy {:.4f}".format(acc))
    

# dataset = 'pubmed'
# if not os.path.exists(f'./exps/{dataset}'): 
#     os.makedirs(f'./exps/{dataset}')
# if rank == 1:
#     np.save('./exps/' + dataset + '/gcn_pp_loss', np.array(loss_list))
#     np.save('./exps/' + dataset + '/gcn_pp_val_acc', np.array(val_acc_list))
#     np.save('./exps/' + dataset + '/gcn_org_loss', np.array(loss_org_list))
#     np.save('./exps/' + dataset + '/gcn_org_val_acc', np.array(val_acc_org_list))
    
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