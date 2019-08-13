# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:15:16 2019

@author: Pavilion
"""

import numpy as np
import random
import math
import sys


sys.path.append('../')
from base import base
from load import load
from util import util
from net_regress import net_regress

rand_pick_exp=1
rand_tslt_range=(0.5,0.5,0.1)
#rand_tslt_range=(0,0,0)
rand_angle_range=(10.0/180*math.pi,10.0/180*math.pi,5.0/180*math.pi)
#rand_angle_range=(0,0,0)
range_rand_fcs=50.0
rand_exp_range=0.3


def gnrt_rot(train_data,one_img,gnrt_num,user,bldshps):
    for ci in range(gnrt_num):
        train_data.append(base.TrainOnePoint(one_img,user))
        train_data[-1].dis=one_img.dis.copy()
        
#here, after rand the dif of angle, we should update slt        
        for i in range(3):        
            train_data[-1].init_angle[i] \
            +=2*rand_angle_range[i]*np.random.random((1,))-rand_angle_range[i]
            
        util.get_slt_land_cor_init(train_data[-1],bldshps,user)
            
def gnrt_tslt(train_data,one_img,gnrt_num,user):
    print('gnrt_tslt')
    for ci in range(gnrt_num):        
        
        train_data.append(base.TrainOnePoint(one_img,user))
        
        for i in range(2):
            train_data[-1].init_tslt[i]\
            +=2*rand_tslt_range[i]*np.random.random((1,))-rand_tslt_range[i] 
        
        train_data[-1].init_tslt[2]\
            +=2*rand_tslt_range[2]*train_data[-1].tslt[2]*np.random.random((1,))-rand_tslt_range[2]*train_data[-1].tslt[2]
        
    
def gnrt_exp(train_data,one_img,gnrt_num,user,all_data,bldshps):
    for ci in range(gnrt_num):        
        train_data.append(base.TrainOnePoint(one_img,user))
        
        if (rand_pick_exp==1):        
            first=random.randrange(len(all_data))
            second=random.randrange(len(all_data[first].data))
            train_data[-1].init_exp=all_data[first].data[second].exp.copy()
        else:
            for i in range(1,train_data[-1].init_exp.shape[0]):                
                train_data[-1].init_exp[i]\
                    +=2*rand_exp_range*np.random.random((1,))-rand_exp_range
                    
        util.get_slt_land_cor_init(train_data[-1],bldshps,user)
                    
                    
def gnrt_fcs(train_data,one_img,gnrt_num,user,bldshps):
    for ci in range(gnrt_num):        
        train_data.append(base.TrainOnePoint(one_img,user))
# =============================================================================
#         print('one_img land cor------------------one_img:\n',one_img.land_cor)
#         print('land cor------------------',train_data[-1].land_cor)
# =============================================================================
        
        train_data[-1].dis=one_img.dis.copy()
                
        train_data[-1].fcs+=(np.random.random((1,))[0])*range_rand_fcs*2-range_rand_fcs
        
        util.recal_dis(train_data[-1],bldshps)

        
def gnrt_user(train_data,one_img,gnrt_num,user,bldshps,all_data):
    for ci in range(gnrt_num):        
        train_data.append(base.TrainOnePoint(one_img,user))
        train_data[-1].dis=one_img.dis.copy()        
        
        first=random.randrange(len(all_data))
        train_data[-1].user=all_data[first].user.copy()
        
        util.recal_dis(train_data[-1],bldshps)
        util.get_slt_land_cor_init(train_data[-1],bldshps,user)

def gnrt_init_dis(one,all_data):
    first=random.randrange(len(all_data))
    second=random.randrange(len(all_data[first].data))
    one.init_dis=all_data[first].data[second].dis.copy()
    


def generate_train_data(data,bldshps):
    
    num_ide=data[0].user.shape[0]
    num_exp=data[0].data[0].exp.shape[0]
    num_land=data[0].data[0].land.shape[0]
    
    assert((num_ide,num_exp)==(bldshps.shape[0],bldshps.shape[1]))    
    
    train_data=[]
    for one_ide in data:
#        spf_bldshps=cal_spf_bldshps(bldshps,one_ide.user)
        for one_img in one_ide.data:
            gnrt_rot(train_data,one_img,5,one_ide.user,bldshps)
            gnrt_tslt(train_data,one_img,5,one_ide.user)
            gnrt_exp(train_data,one_img,15,one_ide.user,data,bldshps)
            gnrt_fcs(train_data,one_img,5,one_ide.user,bldshps)
            gnrt_user(train_data,one_img,5,one_ide.user,bldshps,data)
    
    for x in train_data:
        gnrt_init_dis(x,data)

    
    return train_data

def get_train_ioput(train_data,bldshps):
    
    num_exp=train_data[0].exp.shape[0]-1
    num_land=train_data[0].land.shape[0]
    num_angle=3
    num_tslt=3
    
    mean_ldmk,tri_idx,px_barycenter=load.load_tri_idx('../const_file/tri_idx_px.txt',num_land)    
    num_px=px_barycenter[0].shape[0]
    
    train_data_input=np.empty((len(train_data),num_px))
    train_data_output=np.empty((len(train_data),num_angle+num_tslt+num_exp+num_land*2))
    
    for counter, one_data in enumerate(train_data):
#input:
        init_land=util.get_init_land(one_data,bldshps)
        
# =============================================================================
#         print(init_land[tri_idx[px_barycenter[0]][:,0]].shape)
#         print(tri_idx[px_barycenter[0]][:,0].shape)
#         print(tri_idx[px_barycenter[0]].shape)
#         print(px_barycenter[1][:,0].shape)
#         print(init_land[tri_idx[px_barycenter[0]][:,1]].shape)
# =============================================================================
        
#        t=init_land[tri_idx[px_barycenter[0]][:,0]]*px_barycenter[1][:,0].reshape(num_px,1)
        
#        print(init_land[tri_idx[px_barycenter[0]][:,0]],px_barycenter[1][:,0].reshape(num_px,1),t)
        
        train_data_input[counter]=util.get_input_from_land_img(init_land,one_data.img,tri_idx,px_barycenter)
        
        
#output:
        train_data_output[counter, 0:num_angle]\
            =(one_data.angle-one_data.init_angle).copy()
        train_data_output[counter,num_angle:num_angle+num_tslt]\
            =(one_data.tslt-one_data.init_tslt).copy()
        train_data_output[counter,num_angle+num_tslt:num_angle+num_tslt+num_exp]\
            =(one_data.exp-one_data.init_exp)[1:num_exp+1].copy()
        train_data_output[counter,num_angle+num_tslt+num_exp:num_angle+num_tslt+num_exp+num_land]\
            =(one_data.dis-one_data.init_dis)[:,0].copy()
        train_data_output[counter,num_angle+num_tslt+num_exp+num_land:num_angle+num_tslt+num_exp+num_land*2]\
            =(one_data.dis-one_data.init_dis)[:,1].copy()
        

    return train_data_input,train_data_output

def init_iodata_from_begin():    
    
    
    data,bldshps=load.load_dataHEbldshps()
    #util.get_land(data[0].data[0],bldshps,data[0].user)
    
    train_data=generate_train_data(data,bldshps)
    
    train_data_input,train_data_output=get_train_ioput(train_data,bldshps)
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    print(train_data_input.shape)
    print(train_data_output.shape)
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    np.save('../mid_data/train_data_input_all_7_24',train_data_input)
    np.save('../mid_data/train_data_output_all_7_24',train_data_output)
    
    return train_data_input,train_data_output
    
def init_iodata_from_npy():
    train_data_input=np.load('../../const_file/train_data_input_all_7_24.npy')
    train_data_output=np.load('../../const_file/train_data_output_all_7_24.npy')
#    train_data_input=np.load('../mid_data/train_data_input_all_7_8.npy')
#    train_data_output=np.load('../mid_data/train_data_output_all_7_8.npy')
    return train_data_input,train_data_output
    
    
#train_data_input,train_data_output=init_iodata_from_begin()
train_data_input,train_data_output=init_iodata_from_npy()





net_regress.train_net(train_data_input,train_data_output)
#net_regress.train_net_splt(train_data_input,train_data_output)


