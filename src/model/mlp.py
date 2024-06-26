import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, n_hidden, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, n_hidden)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(n_hidden, output_size)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

    def train_net(self, train_loader, perm=torch.arange(0, 784).long(), n_epoch=1):
        self.train()
        optimizer = torch.optim.AdamW(self.parameters())

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using {device} device")

        self.to(device)

        for epoch in range(n_epoch):
            for step, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                data = data.view(-1, 28*28)
                data = data[:, perm]
                data = data.view(-1, 1, 28, 28)
                
                optimizer.zero_grad()
                logits = self(data)
                loss = F.cross_entropy(logits, target)
                loss.backward()
                optimizer.step()
                
                if step % 100 == 0:
                    print(f"Epoch [{epoch+1}/{n_epoch}], Step [{step}], Loss: {loss.item():.4f}")

    def test(self, test_loader, perm=torch.arange(0, 784).long()):
        self.eval()
        test_loss = 0
        correct = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        with torch.no_grad():
            for step, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                
                data = data.view(-1, 28*28)
                data = data[:, perm]
                data = data.view(-1, 1, 28, 28)
                
                logits = self(data)
                test_loss += F.cross_entropy(logits, target, reduction='sum').item()
                pred = torch.argmax(logits, dim=1)
                correct += pred.eq(target).sum().item()
        
        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        
        print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')
