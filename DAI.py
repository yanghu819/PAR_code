

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
import tqdm as tqdm
import matplotlib.pyplot as plt
from loss_dai import *




A = np.load('label_train_pa.npy')

[m,n] = np.shape(A)
has_weight = 0

print('positive propotion\n', np.sum(A, 0)/m)
print('pp deviate of 0.5\n', np.abs(np.sum(A, 0)/m - 0.5))
print('with e \n', np.exp(np.abs(np.sum(A, 0)/m - 0.5)))
weight = np.exp(np.abs(np.sum(A, 0)/m - 0.5))**3

A = torch.from_numpy(A).float()
y = torch.ones(1,n)*0.5

weight = torch.from_numpy(weight).type(type(y))

if has_weight == 0:
    weight = torch.ones(weight.size())
print(weight)

class Dainet(torch.nn.Module):
    def __init__(self, m, n):
        super(Dainet, self).__init__()
        self.vec_1t_n = torch.ones(1, n)
        self.vec_1t_m = torch.ones(1, m)
        self.m = m
        self.r = Parameter(torch.ones(m, 1))
        
        print(self.r)
        print(self.r.size())
        print(self.r.requires_grad)
        
    def forward(self, x):
        r1t = self.r.mm(self.vec_1t_n) # r1t
        r1tA = r1t.mul(x)
        output = self.vec_1t_m.mm(r1tA)
        
        
        new_m = self.r.t().mm(torch.ones(m, 1))
        return output/new_m
        


net = Dainet(m,n)
optimizer = torch.optim.Adam(net.parameters(), lr = 0.001, weight_decay = 0)

loss_mse = torch.nn.MSELoss()
loss_r = loss_r()
loss_p = loss_p()



for i in range(10000):
     
    output = net(A)
    
    loss_mse_ = loss_mse(output, y) 
    loss_p_ = loss_p(output, weight)
    loss_r_ = loss_r(next(net.parameters()))
    
    loss_all = loss_p_ + 0.001 * loss_r_  

    optimizer.zero_grad()
    loss_all.backward()
    optimizer.step()
    new_m = next(net.parameters()).detach().sum() 
    
    if i % 500 == 0:
        print('%'*30)
        print(i,'new_m', new_m)
        print('mse r p ', loss_mse_.data, loss_p_.data, loss_r_.data)


r_torch = next(net.parameters()).detach()
r_data = r_torch.data.numpy()
print(r_data.shape)
plt.plot(r_data)
plt.show()

r_data[r_data <= 0] = 0
num_n = np.sum(r_data[:,0])
r_data = np.tile(r_data,[1,n])
AA = A.numpy()*(r_data)

print(np.sum(AA,0)/num_n)


for i in r_data[:,1]:
    print(i)

np.sum(r_data[:,0] != 0)

#
