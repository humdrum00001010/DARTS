from torchvision import datasets
from torchvision import transforms
import torch
from model import ResNet

device = torch.device('cuda')
cifar10_val = datasets.CIFAR10('CIFAR-10', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
]))

model = ResNet()
model.load_state_dict(torch.load('nnproject.pt'))
model.to(device=device)
model.eval()

correct = 0
testing_num = 0
for img, label in cifar10_val:
    print("Current Testing no: {}".format(testing_num))
    testing_num += 1
    img = img.to(device=device)
    img = img.unsqueeze(0)
    label = torch.tensor(label).to(device=device)

    with torch.no_grad():
        out = model(img)
        _, predicted = torch.max(out, dim=1)
        correct += int(label == predicted)

print("Accuracy: {}".format(correct / len(cifar10_val)))
