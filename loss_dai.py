

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class loss_r(nn.Module):
    def __init__(self):
        super(loss_r, self).__init__()

    def forward(self, r):
        loss_left = (-5*r)**(1)
#        print('loss_left type', loss_left)
        loss_right = 0.0000 * (r - 1) 
        loss_mid = torch.zeros(loss_right.size())

        loss = torch.max(loss_left, loss_right)
        loss = torch.max(loss, loss_mid)
        loss = torch.sum(loss)
    
        return loss
    
class loss_p(nn.Module):
    def __init__(self):
        super(loss_p, self).__init__()

    def forward(self, p, weight):
        loss_left  = 1.0 * (2 * (0.4 - p))**3
        loss_right = 1.0 * (4*(p - 0.65))**1
        loss_mid = torch.zeros(loss_right.size())

        loss = torch.max(loss_left, loss_right)
        loss = torch.max(loss, loss_mid)
        loss = loss.mul(weight)
#        print('loss_p', loss)
        loss = torch.sum(loss)

    
        return loss 




#
#loss = loss_p()
#
#x_array = np.arange(-100, 100)/100
#x = torch.from_numpy(x_array)
#y_array = loss(x).detach().numpy()
#
#plt.plot(x_array, y_array)
#



















