import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        # super(Classifier, self).__init__() # One of the two
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, X):
        out = F.relu(self.fc1(X))
        out = F.sigmoid(self.fc2(out))
        return out

input_dim = 20
hidden_dim = 64
num_classes = 10

model = Classifier(input_dim, hidden_dim, num_classes)

loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Forward pass example
sample_input = torch.randn(32, input_dim)  # Batch size of 32
outputs = model(sample_input)
loss = loss_fun(outputs, torch.randint(0, num_classes, (32,)))

print(f"Loss: {loss.item()}")



