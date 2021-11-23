import torch
import torch.nn as nn
import torch.nn.functional as f

class neural_model(nn.Module):
    def __init__(self, n_channels):
        super(neural_model, self).__init__()
        #self.bn1 = nn.BatchNorm1d(n_channels, momentum=0.6)
        self.fc1 = nn.Linear(n_channels, 120)
        #self.bn2 = nn.BatchNorm1d(120, momentum=0.6)
        self.fc2 = nn.Linear(120, 80)
        #self.bn3 = nn.BatchNorm1d(80, momentum=0.6)
        self.fc3 = nn.Linear(80, 40)
        #self.bn4 = nn.BatchNorm1d(40, momentum=0.6)
        self.fc4 = nn.Linear(40, 1)
        
    def forward(self, x):
        #x = self.bn1(x)
        x = self.fc1(x)
        x = f.relu(x)
        #x = self.bn2(x)
        x = self.fc2(x)
        x = f.relu(x)
        #x = self.bn3(x)
        x = self.fc3(x)
        x = f.relu(x)
        #x = self.bn4(x)
        x = self.fc4(x)
        x = f.relu(x)
        return x
    
if __name__ == "__main__":
    model = neural_model(80)
    inp = torch.tensor(range(80), dtype=torch.float32)
    model.forward(inp)