import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

TIME_STEP = 10  # rnn多少个时间点
INPUT_SIZE = 1  # 每个时间点多少个数据
LR = 0.02

steps = torch.linspace(0, np.pi * 2, 100, dtype=np.float)
x_np = np.sin(steps)
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label='target (cos)')
plt.plot(steps, x_np, 'b-', label='input (sin)')
plt.legend()
plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        self.linear1 = nn.Linear(32, 1)

    def forward(self, x, h_s):
        # shape
        # x (batch, time_step, input_size)
        # h_s (num_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)
        r_out, h_s = self.rnn(x, h_s)  # input (batch, time_step, input_size)   (h_n, h_c)hidden_state
        # r_out 读完一批数据一个output， 应该选最后一个的输出,最后一个时刻
        outs = []
        # 对于每个时间点，循环
        for time_step in range(r_out.size(1)):
            outs.append(self.linear1(r_out[:, time_step, :]))  # !!!!

        return torch.stack(outs, dim=1), h_s


rnn = RNN()
# print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

h_s = None

for i in range(50):
    start, end = i * np.pi, (i + 1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    prediction, h_s = rnn(x, h_s)

    loss = loss_func(prediction, y)
    print(loss)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)  # !!!!!!!不加会报错
    optimizer.step()

    # print('Epoch: ', epoch, '| Step: ', step, '| loss : ', loss.item(), '| accent : ', accuracy)

# test_output = rnn(test_x[:10].view(-1, 28, 28))
# pred_y = torch.max(test_output, 1)[1].numpy()
# print(pred_y, ' prediction')
# print(test_y[:10], ' real number')
