import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 12, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(12, 16, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 24, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(24, 40, 3, stride=1, padding=1)
        self.conv1x1_1 = nn.Conv2d(3, 12, 1, 1)
        self.conv1x1_2 = nn.Conv2d(12, 24, 1, 1)
        self.linear1 = nn.Linear(2560, 1000)
        self.linear2 = nn.Linear(1000, 200)
        self.linear3 = nn.Linear(200, 10)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.batchnorm2 = nn.BatchNorm2d(12)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.batchnorm4 = nn.BatchNorm2d(24)
        self.batchnorm5 = nn.BatchNorm2d(40)
        """
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        """

    def forward(self, x):
        """"
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        """
        x1 = x
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        temp = x
        x = x + self.conv1x1_1(x1)
        x1 = temp
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.relu(self.batchnorm4(self.conv4(x)))
        x = x + self.conv1x1_2(x1)
        x = self.pool(F.relu(self.batchnorm5(self.conv5(x))))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        logits = self.linear3(x)
        return logits


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform)
    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model = NeuralNetwork()
    batch_size = 4

    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00001)


    def train_loop(dataloader, model, loss_fn, optimizer, losslist):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader, 0):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                losslist.append(loss)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test_loop(dataloader, model, loss_fn, losslist, aclist):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        losslist.append(test_loss)
        correct /= size
        aclist.append(correct)
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    epochs = 10
    losslist = []
    losslist1 = []
    aclist = []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, losslist)
        test_loop(test_dataloader, model, loss_fn, losslist1, aclist)
    print("Done!")
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.plot(aclist)
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(losslist1)
    plt.figure()
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.plot(losslist)
    plt.show()
