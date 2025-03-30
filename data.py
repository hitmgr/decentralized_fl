from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_data():
    """加载MNIST数据集"""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST('data', train=True, download=True, transform=transform)
    testset = datasets.MNIST('data', train=False, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    return trainloader, testloader