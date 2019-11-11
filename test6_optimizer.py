import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as Data  # 进行小batch训练

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

plt.scatter(x.numpy(), y.numpy())
plt.show()

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(1, 20)
        self.hidden2 = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.hidden2(x)
        return x


# 使用不同优化器优化
net_SGD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()

nets = [net_Adam, net_Momentum, net_RMSprop, net_SGD]

opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))

optimizers = [opt_Adam, opt_Momentum, opt_RMSprop, opt_SGD]

loss_func = torch.nn.MSELoss()
losses = [[], [], [], []]  # 损失结合

for i in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader):

        for net, opt, l_his in zip(nets, optimizers, losses):
            prediction = net(batch_x)
            loss = loss_func(prediction, batch_y)
            opt.zero_grad()  # 优化   反向传播
            loss.backward()
            opt.step()
            l_his.append(loss.item())  # loss只有一个参数可以直接.item()提取

lables = ['Adam', 'Momentum', 'RMSprop', 'SGD']

for i, l_his in enumerate(losses):
    plt.plot(l_his, label=lables[i])
plt.legend(loc='best')
plt.xlabel('steps')
plt.ylabel('loss')
plt.ylim((0, 0.2))
plt.show()
