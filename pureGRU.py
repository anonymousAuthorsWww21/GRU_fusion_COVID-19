import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from tqdm import tqdm
import time
import os

# data path
country = 'Austria'
data_dir = 'data/'+country+'_latest.csv'
PATH = 'model/pureGRU_'+country+'.pth'

# Training Setting
lr = 0.0008
hidden_dim = 128
epoch = 15000

# initial
train_x = []
train_y = []
lf = []
k = 0.1#5 or 0.1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#read csv & date format process
df = pd.read_csv(data_dir)
df_res = df.drop('data', axis=1)
df_in = df.loc[:,['D_in', 'R_in', 'A_in']]
data = df_in.values
cks = df_res.values
L = len(data)

obs_days = 29
predict_days = 59
training_days = L- obs_days
# training_days = L

#generate lambda_0
for i in range(len(cks)):
    tmp = 0
    if i+6 >= len(cks):
        lf.append(lf[len(cks)-7])
    else:
        for j in range(5):
            tmp += cks[i+j+1, 0]
        if tmp == 0:
            lf.append(0)
        else:
            lf.append(float(cks[i+6, 0]) / float(tmp))
#generate dataset/dataloader
inputs = np.zeros((1, len(data), df_in.shape[1]))
checks = np.zeros((1, len(cks), df_res.shape[1]))

inputs[0] = data
checks[0] = cks

inputs = inputs.reshape(-1, len(data), df_in.shape[1])
checks = checks.reshape(-1, len(cks), df_res.shape[1])
# inputs = inputs[:, :training_days, :]
# checks = checks[:, :training_days, :]

ground_truth = train_y = inputs[:, 1:, :]
train_x = inputs[:, :training_days-1, :]
train_y = inputs[:, 1:training_days, :]
D_ins = torch.tensor(checks[:, 1:training_days, 3:4])
R_ins = torch.tensor(checks[:, 1:training_days, 4:5])
A_ins = torch.tensor(checks[:, 1:training_days, 5:6])

train_data = torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
train_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=1)

print("Ready.\n")


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, drop_prob=0):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.linear1 = nn.Linear(hidden_dim, 32)
        self.linear2 = nn.Linear(32, 3)
        self.linear3 = nn.Linear(hidden_dim, 3)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.RReLU()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)

        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.d_iq = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.ep = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def forward(self, x, D_ins, R_ins, A_ins, h):
        out, h = self.gru(x, h)
        out = self.linear3(out)
        # out = self.linear2(out)
        # out = self.linear2(out)
        # out = self.relu(out)

        e_1 = self.alpha * x[:, :, 0:1] - A_ins
        e_2 = self.gamma * x[:, :, 0:1] - R_ins
        e_re = (torch.mean(e_1) + torch.mean(e_2)) / 2
        return out, e_re, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        # print(weight)
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        # print(hidden)
        return hidden


def train(train_loader, learn_rate, hidden_dim=hidden_dim, num_epochs=epoch):
    # generate model
    input_dim = next(iter(train_loader))[0].shape[2]
    n_layers = 1
    model = GRUNet(input_dim, hidden_dim, n_layers)

    model.to(device)

    # criterion = My_loss()
    mseloss = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    min_loss = 0.

    # start training
    print('start training!')
    model.train()

    for epoch in range(1, num_epochs + 1):
        # print('Epoch {}/{}'.format(epoch, num_epochs))
        # print('-' * 10)

        # init
        h = model.init_hidden(1)
        avg_loss = 0.

        for x, y in train_loader:
            h = h.data
            optimizer.zero_grad()
            yy, e_re, h = model(x.to(device).float(), D_ins.to(device).float(), R_ins.to(device).float(),
                                A_ins.to(device).float(), h)
            yy = yy.double()
            e_re = e_re.double()
            #loss = mseloss(y.to(device), yy) + e_re
            loss = mseloss(y.to(device), yy)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            '''model.v1.data.clamp_(0)
            model.v2.data.clamp_(0)
            model.v3.data.clamp_(0)
            model.v4.data.clamp_(0)
            model.v5.data.clamp_(0)'''

        if epoch == 1:
            min_loss = avg_loss
        else:
            if avg_loss < min_loss:
                best_state_dict = model.state_dict()
                # torch.save(model.state_dict(), PATH)
                min_loss = avg_loss

        if epoch % 100 == 0:
            print(
                "Epoch {}/{} Done, Total Loss: {}, min Loss: {}".format(epoch, num_epochs, avg_loss / len(train_loader),
                                                                        min_loss))

    print("Complete.\nMin Loss: {}".format(min_loss))
    torch.save(best_state_dict, PATH)
    return model

def test_and_compare():
    input_dim = next(iter(train_loader))[0].shape[2]
    n_layers = 1
    model = GRUNet(input_dim, hidden_dim, n_layers)
    model.load_state_dict(torch.load(PATH))
    # model.load_state_dict(torch.load(PATH))
    model.to(device)
    model.eval()
    print('load best model')

    with torch.no_grad():
        for x, y in train_loader:
            h = model.init_hidden(1)
            h = h.data
            yy, _, h = model(x.to(device).float(), D_ins.to(device).float(), R_ins.to(device).float(), A_ins.to(device).float(), h)

            # h = model.init_hidden(1)
            # h = h.data
            # y_pre = []
            # first_Day = x[:,0:1,:]
            # _ = first_Day
            # for i in range(len(y[0,:,0])):
            #     # freedom
            #     # _, h = model(_.to(device).float(), h)
            #     # N + 1
            #     _, e, h = model(x[:,i:i+1,:].to(device).float(), D_ins[:,i:i+1,:].to(device).float(), R_ins[:,i:i+1,:].to(device).float(), A_ins[:,i:i+1,:].to(device).float(), h)
            #     y_pre.append(_[0,0,:].to('cpu').numpy())
            # y_pre = np.array(y_pre)

            # predict future
            future_D = []
            _ = yy[:,-1:,:]
            for day in range(1, 1+predict_days):
                _, e, h = model(_, 0, 0, 0, h)
                future_D.append(_[0,0,0].to('cpu').numpy())
            future_D = np.array(future_D)
            print(future_D)

            # plot D_ins
            yy = yy.to('cpu')
            plt.plot(ground_truth[0, :, 0], 'grey')
            plt.plot([x for x in range(0, training_days - 1)], yy[0, :, 0], 'r')
            plt.plot([x for x in range(training_days, training_days + predict_days)], future_D, 'x')

            # plot D
            plt.figure(2)
            D = checks[0, :, 0]
            plt.plot(D, 'grey')

            D_verify = [D[0]]
            for x in range(0,training_days-1):
                  D_verify.append(D_verify[-1] + yy[0, x, 0])
            plt.plot([x for x in range(0, training_days)], D_verify, 'r')

            D_predict = [D_verify[-1]]
            for x in range(predict_days):
                 D_predict.append( D_predict[-1] + future_D[x] )
            plt.plot([x for x in range(training_days-1, training_days + predict_days)], D_predict, 'x')

            print(D_verify)
            print('\n')
            print(D_predict)

            plt.show()

#gru_model = train(train_loader, lr)
#time.sleep(5)
test_and_compare()
