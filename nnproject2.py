import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime

device = torch.device('mps')

class MITDataset(Dataset):
    def __init__(self, dirpath, transform=None):
        self.transform = transform
        self.paths = [os.path.join(dirpath, path) for path in os.listdir(dirpath)]
        self.imgs = list(map(lambda x: Image.open(x), self.paths))
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img = self.imgs[index]
        return self.transform(img), torch.tensor(int(self.paths[index].split('/')[-1].split('_')[0]))

train_data = MITDataset('training', transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.2527), std=(0.2044)),
    transforms.Resize((64, 64))
]))
test_data = MITDataset('test', transform=transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4541), std=(0.2720)),
    transforms.Resize((64, 64)),
]))
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)


class Cell(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(num_features=n_channels)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.batchnorm.weight, 0.5)
        torch.nn.init.zeros_(self.batchnorm.bias)
    def forward(self, x):
        out = self.conv(x)
        out = self.batchnorm(out)
        out = torch.relu(out)
        return out + x

class Net(nn.Module):
    def __init__(self, n_channels, n_blocks, classes=10):
        super().__init__()
        self.n_channels = n_channels
        self.color2feature = nn.Conv2d(1, self.n_channels, kernel_size=3, padding=1)
        self.cells = nn.Sequential(*(n_blocks * [Cell(n_channels)]))
        self.fc1 = nn.Linear(16 * 16 * n_channels, classes)
    def forward(self, x):
        out = F.max_pool2d(self.color2feature(x), 2)
        out = F.max_pool2d(self.cells(out), 2)
        out = out.view(-1, 16 * 16 * self.n_channels)
        out = self.fc1(out)
        return out

model = Net(16, 5)
model.to(device=device)
optimizer = optim.SGD(params=model.parameters(), lr=3e-2)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(0, 4):
    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]
        imgs = imgs.to(device=device)
        labels = labels.to(device=device)
        out = model(imgs)
        loss = loss_fn(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("{}: Epoch: {}, Loss: {}".format(datetime.datetime.now(), epoch, loss))

torch.save(model.state_dict(), 'nnproject2.pt')

correct = 0
for img, label in train_data:
    img = img.to(device=device)
    img = img.unsqueeze(0)

    with torch.no_grad():
        out = model(img)
        _, predicted = torch.max(out, dim=1)
        correct += int(label == predicted)

print(correct / len(train_data))