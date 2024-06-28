import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=4,
                               kernel_size=3,
                               padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=4,
                               out_channels=8,
                               kernel_size=3,
                               padding=1)
        self.fc = nn.Linear(8 * 7 * 7, 10)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 8 * 7 * 7)
        return self.fc(x)

    def train_net(self, train_loader, perm=torch.arange(0, 784).long(), n_epoch=1):
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.005)


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
