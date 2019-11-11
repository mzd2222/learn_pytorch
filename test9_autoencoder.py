####  待优化


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

EPOCH = 5
BATCH_SIZE = 64
LR = 0.05

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,  # true的话就给训练数据，false给测试数据
    transform=torchvision.transforms.ToTensor(),  # 改成Tensor 压缩 (0,255)  --->  (0,1)
    download=False
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3)
        )

        self.decode = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # 输出为0-1范围，原始数据范围
        )

    def forward(self, x):
        encode = self.encode(x)
        decode = self.decode(encode)
        return encode, decode


autoEncoder = AutoEncoder()
# print(cnn)

optimizer = torch.optim.SGD(autoEncoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = x.view(-1, 28 * 28)
        b_y = x.view(-1, 28 * 28)

        encoded, decoded = autoEncoder(b_x)

        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('Epoch: ', epoch, '| Step: ', step, '| loss : ', loss.item())

aa1 = train_data.data.clone().detach()[:5].type(torch.float32).view(-1, 28 * 28)

for aa in aa1:
    _, pri = autoEncoder(aa)

    plt.imshow(np.reshape(aa.detach().numpy(), (28, 28)), cmap='gray')
    plt.show()
    plt.imshow(np.reshape(pri.detach().numpy(), (28, 28)), cmap='gray')
    plt.show()
