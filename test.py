import torch
from torchvision import transforms
# from sympy import evaluate
# from torch.optim import optimizer
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x    # 注意别忘记return

    def get_data_loader(self, is_train):
        to_tensor = transforms.Compose([transforms.ToTensor()])
        data_set = MNIST("", is_train, transform=to_tensor, download=True)
        return DataLoader(data_set, batch_size=16, shuffle=True)

    @staticmethod   # 装饰器，转静态方法
    def evaluate(test_data, net):
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            for (x, y) in test_data:
                outputs = net.forward(x.view(-1, 28*28))
                for i, output in enumerate(outputs):
                    if torch.argmax(output) ==y[i]:
                        n_correct += 1
                    n_total += 1
        return n_correct / n_total



def main():
    net = Net()
    train_data = net.get_data_loader(is_train=True)
    test_data = net.get_data_loader(is_train=False)

    print("initial accuracy:", net.evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(2):
        for (x, y) in train_data:
            net.zero_grad()
            output = net.forward(x.view(-1, 28 * 28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "accuracy:", net.evaluate(test_data, net))    # 缩进问题，往后缩进会进入死循环


    for (n, (x, _)) in enumerate(test_data):
        if n > 9:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()


if __name__ == "__main__":
    main()
