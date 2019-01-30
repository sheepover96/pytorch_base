from torchvision import datasets, transforms

train_data = datasets.CIFAR100('./dataset', transform=transforms.Compose([transforms.ToTensor()]), download=True)