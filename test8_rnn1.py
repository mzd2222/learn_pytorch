import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils import data as Data

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28  # rnn多少个时间点
INPUT_SIZE = 28  # 每个时间点多少个数据
LR = 0.01

train_date = dsets.MNIST('./mnist', train=True, transform=transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_date, batch_size=BATCH_SIZE, shuffle=True)

test_data = dsets.MNIST(root='./mnist', train=False)  # 测试数据
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255.0
test_y = test_data.targets.numpy()[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True  # 输入数据维度格式 (batch, time_step, input)
            # batch_first=False,  # 输入数据维度格式 (time_step, batch, input)
        )

        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 10)

    def forward(self, x):
        r_out, s = self.rnn(x)  # input (batch, time_step, input_size)   (h_n, h_c)hidden_state
        # r_out 读完一批数据一个output， 应该选最后一个的输出,最后一个时刻

        out = F.relu(self.linear1(r_out[:, -1, :]))  # (batch, time_step, input_size)
        out = self.linear2(out)

        return out


rnn = RNN()
# print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        x = x.view(-1, 28, 28)

        output = rnn(x)

        loss = loss_func(output, y)

        optimizer.zero_grad()  # 优化   反向传播
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x.view(-1, 28, 28))
            pred_y = torch.max(test_output, 1)[1].numpy()

            accuracy = sum(pred_y == test_y) / len(test_y)

            print('Epoch: ', epoch, '| Step: ', step, '| loss : ', loss.item(), '| accent : ', accuracy)

test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].numpy()
print(pred_y, ' prediction')
print(test_y[:10], ' real number')
