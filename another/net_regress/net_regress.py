# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:49:17 2019

@author: Pavilion
"""

import numpy as np
import torch
import torch.nn.functional
import matplotlib.pyplot as plt

# =============================================================================
# x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
# y=(x**2)+0.2*torch.rand(x.size())
# # y=x.pow(2)+0.2*torch.rand(x.size())
# 
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()
# =============================================================================

class dde_net_fc(torch.nn.Module):
    def __init__(self,n_in,n_hid,n_out):
        super(dde_net_fc, self).__init__()
        self.hidden1=torch.nn.Linear(n_in,n_hid)
        self.hidden2 = torch.nn.Linear(n_hid, n_hid*10)
        self.hidden3 = torch.nn.Linear(n_hid*10, n_hid)
        self.out=torch.nn.Linear(n_hid,n_out)

    def forward(self,x):
        x=self.hidden1(x)
        x=torch.nn.functional.relu(x)
        x = self.hidden2(x)
        x = torch.nn.functional.relu(x)
        x = self.hidden3(x)
        x = torch.nn.functional.relu(x)
        x=self.out(x)
        return x


def train_net(train_data_input,train_data_output):

    train_data_input=train_data_input.astype(np.float32)
    train_data_output=train_data_output.astype(np.float32)
    
    num_input=train_data_input.shape[1]
    num_output=train_data_output.shape[1]
    
    net=dde_net_fc(num_input,num_input*5,num_output)
    # print(net)
    
    optimizer=torch.optim.SGD(net.parameters(),lr=0.0006)
    loss_func=torch.nn.MSELoss()
    #
    
    
    x=torch.tensor(train_data_input)
    y=torch.tensor(train_data_output)
        
    loss_plt=[]
    max_epoch=100
    for epoch in range(max_epoch):
        predict=net.forward(x)
        loss=loss_func(predict,y)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_plt.append(loss.data.numpy())
        print('epoch: %d '% epoch ,'Loss=%.4f'%loss.data.numpy())
# =============================================================================
#         if epoch % 5 ==0:
#             print('epoch: %d '% epoch ,'Loss=%.4f'%loss.data.numpy())
# =============================================================================

            
            
    plt.ion()
    plt.cla()
    plt.scatter(np.linspace(0,max_epoch,num=max_epoch), loss_plt)
    plt.plot(np.linspace(0,max_epoch,num=max_epoch), loss_plt,'r-',lw=1)
    plt.ioff()
    plt.show()

    torch.save(net, '../mid_data/net_7_5_cf3.pkl')  # save entire net
    torch.save(net.state_dict(), '../mid_data/net_7_5_cf3_params.pkl')   # save only the parameters    
