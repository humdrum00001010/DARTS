from operations import OPNAMES
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import datetime
from model import TrainNASNet
import graphviz
from IPython.display import display

device = torch.device('cuda')
data_path = 'CIFAR-10'

cifar10 = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
]))

train_loader = DataLoader(cifar10, batch_size=32, shuffle=True)

model = TrainNASNet(device=device, N=4, L=8, C=32)
if os.path.exists('nnproject.pt'):
    model.load_state_dict(torch.load(
        'nnproject.pt', map_location=torch.device('cpu')))  # why?
model.to(device=device)
optimizer = optim.SGD(params=model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

def storeGraph(names, N=4):
    fp = open("visual.dot", "w")
    fp.write('digraph {\n')
    for i in range(0, N):
        for j in range(0, i + 2):
            arch = names[i * (i + 3) // 2 + j]
            if arch != 'zero':
                src = j - 2
                if src < 0:
                    fp.write('\t\"k{}\"'.format(src))
                else:
                    fp.write('\t\"{}\"'.format(src))
                fp.write('->\"{}\"[label=\"{}\"];\n'.format(i, arch))
        fp.write('\t\"{}\"->\"concat\";\n'.format(i))
    fp.write('\t\"concat\"->\"k\";\n')
    fp.write('}')
    fp.close()

img_no = 0
q = 0
prev_names = []
print(f"Model has been translated to {device}")
for epoch in range(0, 80):
    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]
        imgs = imgs.to(device=device)
        labels = labels.to(device=device)
        if q == 0:
            idxs = torch.argmax(model.alphas[0], dim=1)
            names = list(map(lambda x: OPNAMES[x], idxs))
            if prev_names != names:
                prev_names = names
                storeGraph(names)
                print(names)
                if img_no % 10 == 0:
                    os.system('dot visual.dot -T png > visualO{}.png'.format(img_no))
                img_no += 1

            out = model(imgs)
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            q = 1
        elif q == 1:
            out = model(imgs)
            loss = loss_fn(out, labels)
            with torch.no_grad():
                grad = torch.cat(torch.autograd.grad(loss, model.alphas))
                model.alphas -= (0.03) * grad
            q = 0
    print("{}: Epoch: {}, Loss: {}".format(
        datetime.datetime.now(), epoch, loss))
    if epoch % 20 == 0 and epoch != 0:
        torch.save(model.state_dict(), 'nnproject' +
                   str(epoch) + '.pt')  # backup

torch.save(model.state_dict(), 'nnproject.pt')
