import torch
import sys
from od_execution.od_execution import od_execution_wrapper
from od_execution.client import send_signal
from od_execution.timer import Timer
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import threading
import os


args = None

def nvtx_mark(title):
    import torch.cuda.nvtx as nvtx
    nvtx.mark(title)


def pause_notifier():
    import time
    time.sleep(10)
    import os
    print("send pause")
    pid = os.getpid()
    nvtx_mark("pause")
    send_signal(pid, "pause")

def resume_notifier():
    import time
    time.sleep(15)
    import os
    print("send resume")
    nvtx_mark("resume")
    pid = os.getpid()
    send_signal(pid, "resume")


def set_notifiers():
    t1 = threading.Thread(target=pause_notifier)
    # t2 = threading.Thread(target=resume_notifier)
    t1.start()
    # t2.start()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.timer = Timer()

    def forward(self, x):
        self.timer.start()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        self.timer.stop()
        print(f"forward time: {self.timer.elapsed(reset=True)}ms")
        return output


# 2. 定义 MLP 模型
class LargeMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5):
        super(LargeMLP, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # 添加 BatchNorm
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # 添加 Dropout
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))  # 输出层
        self.network = nn.Sequential(*layers)
        self.timer = Timer()

    def forward(self, x):
        self.timer.start()
        x = x.view(x.size(0), -1)  # 展平成一维输入
        x = self.network(x)
        self.timer.stop()
        print(f"forward time: {self.timer.elapsed(reset=True)}ms")
        return x



class LargeCNN(nn.Module):
    def __init__(self):
        super(LargeCNN, self).__init__()
        
        # 增加卷积层通道数
        self.conv1 = nn.Conv2d(1, 64, 3, 1)  # 64 filters
        self.conv2 = nn.Conv2d(64, 256, 3, 1) # 128 filters
        self.conv3 = nn.Conv2d(256, 256, 3, 1) # 新增卷积层
        self.conv4 = nn.Conv2d(256, 512, 3, 1) # 新增卷积层
        
        # 增加全连接层大小
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(512 * 10 * 10, 4096)  # 4096神经元
        self.fc2 = nn.Linear(4096, 1024)  # 额外的全连接层
        self.fc3 = nn.Linear(1024, 10)  # 输出层
        self.timer = Timer()
        
    def forward(self, x):
        self.timer.start()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = F.max_pool2d(x, 2)  # 池化降低计算量
        x = torch.flatten(x, 1)  # 展平
        x = self.dropout1(x)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.dropout2(x)
        x = self.fc3(x)  # 输出
        output = F.log_softmax(x, dim=1)
        self.timer.stop()
        print(f"forward time: {self.timer.elapsed(reset=True)} ms")
        return x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    target = torch.LongTensor(args.batch_size).random_(10).to(device)
    for batch_idx, images in enumerate(train_loader):
        # data, target = data.to(device), target.to(device)
        data = images.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    global args
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=4000, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    print(os.getpid())
    # set_notifiers()

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])
    # dataset1 = datasets.MNIST('../data', train=True, download=True,
    #                    transform=transform)
    # dataset2 = datasets.MNIST('../data', train=False,
    #                    transform=transform)
    # train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    # test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    from torch.utils.data import Dataset, DataLoader
    class RandomDataset(Dataset):
        def __init__(self, length):
            self.len = length
            self.data = torch.randn(1, 28, 28, length)

        def __getitem__(self, index):
            return self.data[:, :, :, index]

        def __len__(self):
            return self.len
    train_dataset = RandomDataset(args.batch_size * args.iterations)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)

    # # 定义模型
    # input_size = 1 * 28 * 28  # CIFAR-10 图片展开后的维度
    # hidden_sizes = [16384, 8192, 4096, 4096, 4096, 1024, 512, 128]  # 深度 MLP
    # output_size = 10  # CIFAR-10 有 10 类
    # dropout_rate = 0.3

    # model = LargeMLP(input_size, hidden_sizes, output_size, dropout_rate).to(device)
    model = Net().to(device)
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    model = od_execution_wrapper(model)

    # pp_profile = torch.profiler.profile
    # with pp_profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     schedule=torch.profiler.schedule(
    #         skip_first=80, wait=10, warmup=10, active=150, repeat=1
    #     ),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(
    #         f"./"
    #     ),
    #     with_stack=True,
    #     with_modules=True,
    #     profile_memory=True,
    # ) as prof:
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        # test(model, device, test_loader)
        scheduler.step()
            # prof.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()


