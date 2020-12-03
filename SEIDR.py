import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = "../data/Italy.csv"
PATH = "../model/model_Italy.pth"
lr = 0.00001
num = 500000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(data_dir)

df_A_in_next = df.drop('R_in_next', axis=1).drop('D_in_next', axis=1).drop('A', axis=1).drop('R', axis=1).drop('D', axis=1).drop('A_in', axis=1).drop('R_in', axis=1).drop('D_in', axis=1).drop('data', axis=1)
df_R_in_next = df.drop('A_in_next', axis=1).drop('D_in_next', axis=1).drop('A', axis=1).drop('R', axis=1).drop('D', axis=1).drop('A_in', axis=1).drop('R_in', axis=1).drop('D_in', axis=1).drop('data', axis=1)
df_D_in_next = df.drop('R_in_next', axis=1).drop('A_in_next', axis=1).drop('A', axis=1).drop('R', axis=1).drop('D', axis=1).drop('A_in', axis=1).drop('R_in', axis=1).drop('D_in', axis=1).drop('data', axis=1)
df_A_in = df.drop('R_in', axis=1).drop('D_in', axis=1).drop('A', axis=1).drop('R', axis=1).drop('D', axis=1).drop('A_in_next', axis=1).drop('R_in_next', axis=1).drop('D_in_next', axis=1).drop('data', axis=1)
df_R_in = df.drop('A_in', axis=1).drop('D_in', axis=1).drop('A', axis=1).drop('R', axis=1).drop('D', axis=1).drop('A_in_next', axis=1).drop('R_in_next', axis=1).drop('D_in_next', axis=1).drop('data', axis=1)
df_D_in = df.drop('R_in', axis=1).drop('A_in', axis=1).drop('A', axis=1).drop('R', axis=1).drop('D', axis=1).drop('A_in_next', axis=1).drop('R_in_next', axis=1).drop('D_in_next', axis=1).drop('data', axis=1)
df_D = df.drop('R_in_next', axis=1).drop('D_in_next', axis=1).drop('A_in_next', axis=1).drop('R', axis=1).drop('A', axis=1).drop('A_in', axis=1).drop('R_in', axis=1).drop('D_in', axis=1).drop('data', axis=1)
df_R = df.drop('R_in_next', axis=1).drop('D_in_next', axis=1).drop('A_in_next', axis=1).drop('D', axis=1).drop('A', axis=1).drop('A_in', axis=1).drop('R_in', axis=1).drop('D_in', axis=1).drop('data', axis=1)
df_A = df.drop('R_in_next', axis=1).drop('D_in_next', axis=1).drop('A_in_next', axis=1).drop('R', axis=1).drop('D', axis=1).drop('A_in', axis=1).drop('R_in', axis=1).drop('D_in', axis=1).drop('data', axis=1)

A_in_next_set = df_A_in_next.values
R_in_next_set = df_R_in_next.values
D_in_next_set = df_D_in_next.values
A_in_set = df_A_in.values
R_in_set = df_R_in.values
D_in_set = df_D_in.values
D_set = df_D.values
R_set = df_R.values
A_set = df_A.values

L = len(A_in_next_set)

lf = []
for i in range(L):
    tmp = 0
    if i+6 >= L:
        lf.append(lf[L-7])
    else:
        for j in range(5):
            tmp += D_set[i+j+1]
        if tmp == 0:
            lf.append(0)
        else:
            lf.append(float(D_set[i+6])/float(tmp))

A_in_next_set_np = A_in_next_set.ravel()
R_in_next_set_np = R_in_next_set.ravel()
D_in_next_set_np = D_in_next_set.ravel()
A_in_set_np = A_in_set.ravel()
R_in_set_np = R_in_set.ravel()
D_in_set_np = D_in_set.ravel()
D_set_np = D_set.ravel()
R_set_np = R_set.ravel()
A_set_np = A_set.ravel()

D_0 = torch.Tensor([[[D_set_np[0]]]])
D_i_0 = torch.Tensor([[[D_in_next_set_np[0]]]])
D_i_1 = torch.Tensor([[[D_in_next_set_np[1]]]])
R_0 = torch.Tensor([[[R_set_np[0]]]])
R_i_0 = torch.Tensor([[[R_in_next_set_np[0]]]])
R_i_1 = torch.Tensor([[[R_in_next_set_np[1]]]])
A_0 = torch.Tensor([[[A_set_np[0]]]])
A_i_0 = torch.Tensor([[[A_in_next_set_np[0]]]])
A_i_1 = torch.Tensor([[[A_in_next_set_np[1]]]])

D_actual = []
R_actual = []
A_actual = []
for i in range(L):
    D_actual.append(torch.Tensor([[[D_set_np[i]]]]).to(device))
    R_actual.append(torch.Tensor([[[R_set_np[i]]]]).to(device))
    A_actual.append(torch.Tensor([[[A_set_np[i]]]]).to(device))

x_axis = []
for i in range(L):
    x_axis.append(i+1)

print("Ready.\n")


class SEIDR(nn.Module):
    def __init__(self):
        super(SEIDR, self).__init__()
        self.af = nn.Parameter(torch.Tensor([0.0067]), requires_grad=True)
        self.gm = nn.Parameter(torch.Tensor([0.0274]), requires_grad=True)
        self.ep = nn.Parameter(torch.Tensor([0.97]), requires_grad=True)
        self.di = nn.Parameter(torch.Tensor([0.36]), requires_grad=True)
        self.k = nn.Parameter(torch.Tensor([1.6]), requires_grad=True)

    def forward(self, D_0, D_i_0, D_i_1, R_0, R_i_0, R_i_1, A_0, A_i_0, A_i_1):
        D = [D_0]
        D_i = [D_i_0]
        R = [R_0]
        R_i = [R_i_0]
        A = [A_0]
        A_i = [A_i_0]

        I = []
        I_0_tmp = (D_i_0 + R_i_0 + A_i_0).float() / self.di.float()
        I_1_tmp = (D_i_1 + R_i_1 + A_i_1).float() / self.di.float()
        I.append(I_0_tmp)

        E = []
        E.append(((I_1_tmp - I_0_tmp) + self.di * I_0_tmp) / self.ep)

        for i in range(L - 1):
            D_t = D[len(D) - 1]
            D_i_t = D_i[len(D_i) - 1]
            D_t_1 = D_t + D_i_t
            D.append(D_t_1)

            R_t = R[len(R) - 1]
            R_i_t = R_i[len(R_i) - 1]
            R_t_1 = R_t + R_i_t
            R.append(R_t_1)

            A_t = A[len(A) - 1]
            A_i_t = A_i[len(A_i) - 1]
            A_t_1 = A_t + A_i_t
            A.append(A_t_1)

            I_t = I[len(I) - 1]
            E_t = E[len(E) - 1]
            I_i_t = self.ep * E_t - self.di * I_t
            I_t_1 = I_t + I_i_t
            I.append(I_t_1)

            R_i_t_1 = self.gm * D_t_1
            A_i_t_1 = self.af * D_t_1
            D_i_t_1 = self.di * I_t_1 - (self.gm + self.af) * D_t_1
            R_i.append(R_i_t_1)
            A_i.append(A_i_t_1)
            D_i.append(D_i_t_1)

            E_i_t = lf[i] * (self.k * E_t + I_t) - self.ep * E_t
            E_t_1 = E_t + E_i_t
            E.append(E_t_1)

        print("af={}, gm={}, ep={}, di={}, k={}".format(self.af.item(), self.gm.item(), self.ep.item(), self.di.item(), self.k.item()))
        print('-' * 10)
        return D, R, A


def train(learn_rate, num_epochs):
    model = SEIDR()
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    min_loss = 0.

    model.train()
    plt.figure()
    plt.ion()

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        tot_loss = 0.

        optimizer.zero_grad()
        D_pred, R_pred, A_pred= model(D_0.to(device).float(), D_i_0.to(device).float(), D_i_1.to(device).float(), R_0.to(device).float(), R_i_0.to(device).float(), R_i_1.to(device).float(), A_0.to(device).float(), A_i_0.to(device).float(), A_i_1.to(device).float())

        loss = criterion(D_pred[0], D_actual[0]) + criterion(R_pred[0], R_actual[0]) + criterion(A_pred[0], A_actual[0])
        for i in range(1, L):
            loss = loss + criterion(D_pred[i], D_actual[i]) + criterion(R_pred[i], R_actual[i]) + criterion(A_pred[i], A_actual[i])

        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

        D_draw = []
        for i in range(L):
            D_draw.append(D_pred[i].item())

        plt.cla()
        plt.plot(x_axis, D_set_np, "x", color="r", label="Actual")
        plt.plot(x_axis, D_draw, color="b", label="active cases (Predicted)")
        plt.draw()
        plt.pause(0.01)

        if epoch == 1:
            min_loss = tot_loss
        else:
            if tot_loss < min_loss:
                torch.save(model.state_dict(), PATH)
                min_loss = tot_loss

        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, num_epochs, tot_loss))
        print('-' * 10)

        # new_lr = learn_rate * (0.1 ** (epoch // 5000))
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = new_lr

    plt.ioff()
    plt.show()

    print("Complete.\nMin Loss: {}".format(min_loss))


train(lr, num)
