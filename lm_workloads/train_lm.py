import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import random

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 数据加载与预处理
tokenizer = get_tokenizer("basic_english")

# 读取IMDB数据集
train_iter, test_iter = IMDB(split=('train', 'test'))

# 构建词汇表
def yield_tokens(data_iter):
    for label, line in data_iter:
        yield tokenizer(line)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<PAD>", "<UNK>"])
vocab.set_default_index(vocab["<UNK>"])

# 转换文本为张量
def text_pipeline(text):
    return vocab(tokenizer(text))

def label_pipeline(label):
    return 1 if label == 'pos' else 0

# 自定义Dataset
class IMDBDataset(Dataset):
    def __init__(self, data):
        self.data = list(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, text = self.data[idx]
        text_tensor = text_pipeline(text)
        label_tensor = label_pipeline(label)
        return torch.tensor(text_tensor, dtype=torch.long), torch.tensor(label_tensor, dtype=torch.long)

# 重新加载数据
train_iter, test_iter = IMDB(split=('train', 'test'))
train_data = IMDBDataset(train_iter)
test_data = IMDBDataset(test_iter)

# 数据填充与批处理
def collate_batch(batch):
    texts, labels = zip(*batch)
    max_len = max(len(text) for text in texts)
    texts = [text + [vocab["<PAD>"]] * (max_len - len(text)) for text in texts]
    return torch.tensor(texts, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

# 创建 DataLoader
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# 2. 定义LSTM分类模型
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<PAD>"])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, text):
        embedded = self.embedding(text)  # (batch, seq_len, embed_dim)
        _, (hidden, _) = self.lstm(embedded)  # hidden: (num_layers, batch, hidden_dim)
        return self.fc(hidden[-1])  # 取最后一层LSTM的hidden state作为分类输入

# 超参数
vocab_size = len(vocab)
embed_dim = 100
hidden_dim = 128
output_dim = 2

# 实例化模型
model = TextClassificationModel(vocab_size, embed_dim, hidden_dim, output_dim).to(device)

# 3. 训练模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}")

# 4. 测试模型
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for texts, labels in test_loader:
        texts, labels = texts.to(device), labels.to(device)
        outputs = model(texts)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct / total:.4f}")
