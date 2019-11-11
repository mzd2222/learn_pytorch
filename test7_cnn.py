import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision

EPOCH = 1
BATCH_SIZE = 5
LR = 0.05

print(torch.cuda.is_available())

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,  # true的话就给训练数据，false给测试数据
    transform=torchvision.transforms.ToTensor(),  # 改成Tensor 压缩 (0,255)  --->  (0,1)
    download=False
)
test_data = torchvision.datasets.MNIST(root='./mnist', train=False)  # 测试数据

print(train_data.data.size())
print(train_data.targets.size())
plt.imshow(train_data.data[0], cmap='gray')
plt.title('%i' % train_data.targets[0])
plt.show()

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255.0
test_y = test_data.targets[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            # stride跳度
            # padding = (kernel_size-1)/2 保证与原图片等大
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))

        self.lin1 = nn.Linear(32 * 7 * 7, 64)
        self.lin2 = nn.Linear(64, 10)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)  # (batch, 32, 7, 7) ----->>>  (batch, 32*7*7)  !!!!!!!!
        x = F.relu(self.lin1(x))
        out = self.lin2(x)

        return out


cnn = CNN().cuda()
# print(cnn)

optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

cnn.cuda()
loss_func = loss_func.cuda()

for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        output = cnn(batch_x)

        loss = loss_func(output, batch_y)

        optimizer.zero_grad()  # 优化   反向传播
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            test_x = test_x.cuda()
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].cpu()

            accuracy = sum(pred_y.numpy() == test_y.numpy()) / len(test_y.numpy())

            print('Epoch: ', epoch, '| Step: ', step, '| loss : ', loss.item(), '| accent : ', accuracy)

test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].cpu().numpy()
print(pred_y, ' prediction')
print(test_y.numpy()[:10], ' real number')

torch.save(cnn, './model/cnn.pkl')  # 保存整个神经网络
torch.save(cnn.state_dict(), './model/cnn_param.pkl')  # 只保存所有参数
