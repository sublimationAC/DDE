# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:49:17 2019

@author: Pavilion
"""

import numpy as np
import torch
import torch.nn.functional
import matplotlib.pyplot as plt
import torch.utils.data as Data
import os
import math
import random


from load import load
from util import util
from base import base

rand_pick_exp=1
rand_tslt_range=(0.5,0.5,0.1)
#rand_tslt_range=(0,0,0)
rand_angle_range=(10.0/180*math.pi,10.0/180*math.pi,5.0/180*math.pi)
#rand_angle_range=(0,0,0)
range_rand_fcs=50.0
rand_exp_range=0.3


max_cnt=20

def net_test_img_stc(all_data,net,bldshps,mean_ldmk,tri_idx,px_barycenter):
    
    def gen_rand_data(data,all_data):
        for i in range(3):        
            data.angle[i]+=2*rand_angle_range[i]*np.random.random((1,))-rand_angle_range[i]
            
        for i in range(2):
            data.tslt[i]\
                +=2*rand_tslt_range[i]*np.random.random((1,))-rand_tslt_range[i]         
        data.tslt[2]\
            +=2*rand_tslt_range[2]*data.tslt[2]*np.random.random((1,))-rand_tslt_range[2]*data.tslt[2]
        
        for i in range(1,data.exp.shape[0]):
            data.exp[i]\
                +=2*rand_exp_range*np.random.random((1,))-rand_exp_range
# =============================================================================
#         first=random.randrange(len(all_data))
#         second=random.randrange(len(all_data[first].data))
#         data.exp[i]=all_data[first].data[second].exp.copy()
# =============================================================================
        
        first=random.randrange(len(all_data))
        second=random.randrange(len(all_data[first].data))
        data.dis=all_data[first].data[second].dis.copy()
            
    
    num_exp=bldshps.shape[1]-1
    num_land=all_data[0].data[0].land.shape[0]
    num_angle=3
    num_tslt=3
    #util.get_land(data[0].data[0],bldshps,data[0].user)
    
    ITE_num=100
    
    
    

    cnt=np.zeros((max_cnt),int)
    
    for ppp in range(100):
        first=random.randrange(len(all_data))
        second=random.randrange(len(all_data[first].data))
        
        user=all_data[first].user
        one_img=all_data[first].data[second]
        spf_bldshps=np.tensordot(bldshps,user,axes=(0,0))
        
        for ite in range(ITE_num):
            test_data=base.TestOnePoint(one_img,user)
            std_land=test_data.land
            gen_rand_data(test_data,all_data)
    #                bf_land=util.get_land(test_data,bldshps,one_ide.user)
            bf_land=util.get_land_spfbldshps(test_data,spf_bldshps)
            data_input=util.get_input_from_land_img(bf_land,test_data.img,tri_idx,px_barycenter)
                            
            out=net(torch.tensor(data_input.astype(np.float32)).cuda()).cpu()
            out_data=out.data.numpy()
            angle,tslt,exp=out_data[0:num_angle],out_data[num_angle:num_angle+num_tslt],out_data[num_angle+num_tslt:num_angle+num_tslt+num_exp]
            dis=np.empty((num_land,2))
            dis[:,0]=out_data[num_angle+num_tslt+num_exp:num_angle+num_tslt+num_exp+num_land]
            dis[:,1]=out_data[num_angle+num_tslt+num_exp+num_land:num_angle+num_tslt+num_exp+num_land*2]
    
            test_data.angle+=angle
            test_data.tslt+=tslt
            test_data.exp[1:num_exp+1]+=exp
            test_data.dis+=dis
            
    #                aft_land=util.get_land(test_data,bldshps,one_ide.user)
            aft_land=util.get_land_spfbldshps(test_data,spf_bldshps)
            
            rate=\
                math.sqrt((np.linalg.norm(aft_land-std_land)**2)/num_land)\
                /math.sqrt((np.linalg.norm(bf_land-std_land)**2)/num_land)
            
            p=int(rate*10)
            if (p>=max_cnt):
                cnt[-1]+=1
            else:
                cnt[p]+=1
                    
    return cnt




class dde_net_fc(torch.nn.Module):
    def __init__(self,n_in,n_hid,n_out):
        super(dde_net_fc, self).__init__()
        self.hidden1=torch.nn.Linear(n_in,n_hid)
        self.hidden2 = torch.nn.Linear(n_hid, n_hid)
        self.hidden3 = torch.nn.Linear(n_hid, n_hid)
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
    os.environ["CUDA_VISIBLE_DEVICES"]='3'
    LR=0.5
    BATCH_SIZE=10000
    max_epoch=20000
    test_epoch_num=400
    hid_mul=2
    server='104'
    file_num=6
    STEP_SIZE=2000
    GAMMA=0.5
    suffix='_7_24_cf3_f'+str(file_num)+'_lr'+str(LR*100)+'_bch'+\
            str(BATCH_SIZE/10000)+'w_hid'+str(hid_mul)+'_it'+str(max_epoch/10000)+'w_'+server
    
    
    train_data_input=train_data_input.astype(np.float32)
    train_data_output=train_data_output.astype(np.float32)
    
    num_input=train_data_input.shape[1]
    num_output=train_data_output.shape[1]

    
    print(torch.cuda.device_count())
#    torch.cuda.set_device(6,7)
    
    net=dde_net_fc(num_input,num_input*hid_mul,num_output).cuda()
    # print(net)
    
    
    optimizer=torch.optim.SGD(net.parameters(),lr=LR)
#    optimizer= torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
    loss_func=torch.nn.MSELoss()
    #
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    
# =============================================================================
#     x=torch.tensor(train_data_input)
#     y=torch.tensor(train_data_output)
# =============================================================================

    train_data_num=train_data_input.shape[0]  #10000
    x=torch.Tensor(train_data_input[:train_data_num]).cuda()
    y=torch.Tensor(train_data_output[:train_data_num]).cuda()
    
    torch_dataset = Data.TensorDataset(x, y)
    
    train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=6)
    
    
    data,bldshps=load.load_dataHEbldshps()
    mean_ldmk,tri_idx,px_barycenter=load.load_tri_idx('../../const_file/tri_idx_px.txt',data[0].data[0].land.shape[0])

    loss_plt=[]
    
    
    test_cnt=np.empty((max_epoch//test_epoch_num,max_cnt))
    
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
        scheduler.step()
        if epoch %test_epoch_num==0:
            test_cnt[epoch//test_epoch_num]=net_test_img_stc(data,net,bldshps,mean_ldmk,tri_idx,px_barycenter)
            rate=test_cnt[epoch//test_epoch_num,:10].sum()/test_cnt[epoch//test_epoch_num].sum()
            print('epoch: %d '% epoch ,
                  'test <1 rate=%.4f'%rate)    

    
    torch.save(net, '../mid_data/net'+suffix+'.pkl')  # save entire net
#    torch.save(net.state_dict(), '../mid_data/net_7_17_cf3_params_norm_gpu.pkl')   # save only the parameters            
    
    np.save('../mid_data/test_cnt'+suffix,test_cnt)
    
    print('final test <1 rate=%.4f'%(test_cnt[-1,:10].sum()/test_cnt[-1].sum())
            ,'  <0.5=%.4f'%(test_cnt[-1,:5].sum()/test_cnt[-1].sum()))   
    
    y=test_cnt[:,:10].sum(1)        
    plt.ion()
    plt.cla()
    plt.figure(file_num, figsize=(16, 12))
    
    plt.subplot(221)
    plt.scatter(np.linspace(0,len(loss_plt),num=len(loss_plt)), loss_plt)
    plt.plot(np.linspace(0,len(loss_plt),num=len(loss_plt)), loss_plt,'r-',lw=1, label='loss')
    plt.legend(loc='best')
    
    plt.subplot(222)
    plt.scatter(np.linspace(0,len(y),num=len(y)), y)
    plt.plot(np.linspace(0,len(y),num=len(y)), y,'r-',lw=1, label='<1')
    plt.legend(loc='best')
    
    plt.subplot(223)
    plt.scatter(np.linspace(0,len(y),num=len(y)), test_cnt[:,:5].sum(1))
    plt.plot(np.linspace(0,len(y),num=len(y)), test_cnt[:,:5].sum(1),'r-',lw=1, label='<0.5')
    plt.legend(loc='best')
    
    plt.subplot(224)
    plt.scatter(np.linspace(0,len(y),num=len(y)), test_cnt[:,:10].sum(1) )
    plt.plot(np.linspace(0,len(y),num=len(y)), test_cnt[:,:10].sum(1) ,'r-',lw=1)
    plt.scatter(np.linspace(0,len(y),num=len(y)), test_cnt[:,:8].sum(1) )
    plt.plot(np.linspace(0,len(y),num=len(y)), test_cnt[:,:8].sum(1) ,'b-',lw=1)
    plt.scatter(np.linspace(0,len(y),num=len(y)), test_cnt[:,:5].sum(1) )
    plt.plot(np.linspace(0,len(y),num=len(y)), test_cnt[:,:5].sum(1) ,'y-',lw=1)
    plt.scatter(np.linspace(0,len(y),num=len(y)), test_cnt[:,:3].sum(1) )
    plt.plot(np.linspace(0,len(y),num=len(y)), test_cnt[:,:3].sum(1) ,'g-',lw=1)
    
    plt.ioff()
    plt.show()