# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:54:58 2019

@author: Pavilion
"""
import torch
import torch.nn.functional
import torch.nn as nn
import torch.utils.data as Data
import os
import matplotlib.pyplot as plt
import numpy as np

class CNN(nn.Module):
    def __init__(self,hid_n,out_n):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=9, kernel_size=5, stride=1, padding=2,),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=2),)
        self.conv2 = nn.Sequential(nn.Conv2d(9, 18, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(5),)
        self.hidden = nn.Linear(12*23*23, hid_n)
        self.out = nn.Linear(hid_n, out_n)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)                
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = torch.nn.functional.relu(x)
        output = self.out(x)
        return output

def train_smp_cnn(ipt_dt,opt_dt):
    
    os.environ["CUDA_VISIBLE_DEVICES"]='3'
    print(torch.cuda.device_count())
#    torch.cuda.set_device(6,7)
    num_output=opt_dt.shape[1]
    num_hid=num_output*2
    
    
    net=CNN(num_hid,num_output).cuda()
    # print(net)
    
    LR=0.03
    optimizer=torch.optim.SGD(net.parameters(),lr=LR)
#    optimizer= torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
    loss_func=torch.nn.MSELoss()
    #    
    
    x=torch.Tensor(ipt_dt).cuda()
    y=torch.Tensor(opt_dt).cuda()
    
    torch_dataset = Data.TensorDataset(x, y)
    BATCH_SIZE=100
    train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    
    
    loss_plt=[]
    max_epoch=2000
    for epoch in range(max_epoch):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            
            predict=net.forward(batch_x)
            loss=loss_func(predict,batch_y)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            loss_plt.append(loss.data.cpu().numpy())
            print('epoch: %d '% epoch ,'step: %d '% step,'Loss=%.4f'%loss_plt[-1])
# =============================================================================
#         if epoch % 5 ==0:
#             print('epoch: %d '% epoch ,'Loss=%.4f'%loss.data.numpy())
# =============================================================================

    torch.save(net, '../mid_data/net_7_18_smpcnn.pkl')  # save entire net
    torch.save(net.state_dict(), '../mid_data/net_7_18_smpcnn_params.pkl')   # save only the parameters            
            
    plt.ion()
    plt.cla()
    plt.scatter(np.linspace(0,len(loss_plt),num=len(loss_plt)), loss_plt)
    plt.plot(np.linspace(0,len(loss_plt),num=len(loss_plt)), loss_plt,'r-',lw=1)
    plt.ioff()
    plt.show()


