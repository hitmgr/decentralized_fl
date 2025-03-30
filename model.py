import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(10 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 10 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(source, model_class, map_location=None):
    model = model_class()
    if isinstance(source, str):
        state_dict = torch.load(source, map_location=map_location)
    else:
        state_dict = torch.load(source, map_location=map_location)
    model.load_state_dict(state_dict)
    return model

if __name__ == "__main__":
    model = CNN()
    save_model(model, "ipfs_models/initial_model.pth")
    print("初始模型已生成并保存为 initial_model.pth")