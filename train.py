import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from resnet_model import resnet18

GPU = torch.cuda.is_available()


def train(epoch, model, device, train_loader, optimizer):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
    
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    print('accuracy=',100. * correct/len(test_loader.dataset))


def main():
    device = torch.device("cuda" if GPU else "cpu")
    model = resnet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_data = torchvision.datasets.CIFAR100('./dataset',
                        transform=transforms.Compose([transforms.ToTensor()]), download=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_data = torchvision.datasets.CIFAR100('./dataset',
                        transform=transforms.Compose([transforms.ToTensor()]),  train=False, download=True)
    test_loader = torch.utils.data.DataLoader(test_data)


    for epoch in range(100):
        train(epoch, model, device, train_loader, optimizer)
        test(model, device, test_loader)
    


if __name__ == '__main__':
    main()