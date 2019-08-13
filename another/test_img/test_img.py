# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 11:22:44 2019

@author: Pavilion
"""

import numpy as np
import torch
import copy
import cv2
import sys
import random
import math

sys.path.append('../')
from train import train
from load import load
from util import util
from base import base

model_suffix='_7_17_cf3_norm_gpu_1w_hid5_bch1w_lr1'
    
def test_set_img_show(data,net):
    
    bldshps_path='/home/weiliu/fitting_dde/const_file/deal_data/blendshape_ide_svd_77.lv'
    bldshps=load.load_bldshps(bldshps_path,num_ide=77,num_exp=47,num_vtx=11510)
    util.bld_reduce_neu(bldshps)
    
    num_exp=bldshps.shape[1]-1
    num_land=data[0].data[0].land.shape[0]
    num_angle=3
    num_tslt=3
    #util.get_land(data[0].data[0],bldshps,data[0].user)
    
    test_data=train.generate_train_data(data,bldshps)
    
    test_data_input,test_data_output=train.get_train_ioput(test_data,bldshps)
    
    test_data_input=test_data_input.astype(np.float32)
    
    
    for x,y,z in zip(test_data_input,test_data_output,test_data):
        print('A')
        init_land=util.get_init_land(z,bldshps)
        print('B')
        out=net(torch.tensor(x))
        
        print(out.data.numpy()-y)
        
        dif=out.data.numpy()-y
        
        angle,tslt,exp=dif[0:num_angle],dif[num_angle:num_angle+num_tslt],dif[num_angle+num_tslt:num_angle+num_tslt+num_exp]
        dis=np.empty((num_land,2))
        dis[:,0]=dif[num_angle+num_tslt+num_exp:num_angle+num_tslt+num_exp+num_land]
        dis[:,1]=dif[num_angle+num_tslt+num_exp+num_land:num_angle+num_tslt+num_exp+num_land*2]
        
        print('angle dif:  norm:', angle,np.linalg.norm(angle))
        print('tslt dif:  norm:', tslt,np.linalg.norm(tslt))
        print('exp dif:  norm:', exp,np.linalg.norm(exp))
        
        print('dis dif:  norm:', dis,np.linalg.norm(dis))
    
        out_data=out.data.numpy()
        angle,tslt,exp=out_data[0:num_angle],out_data[num_angle:num_angle+num_tslt],out_data[num_angle+num_tslt:num_angle+num_tslt+num_exp]
        dis=np.empty((num_land,2))
        dis[:,0]=out_data[num_angle+num_tslt+num_exp:num_angle+num_tslt+num_exp+num_land]
        dis[:,1]=out_data[num_angle+num_tslt+num_exp+num_land:num_angle+num_tslt+num_exp+num_land*2]
        aft_dt=copy.deepcopy(z)
        aft_dt.init_angle+=angle
        aft_dt.init_tslt+=tslt
        aft_dt.init_exp[1:num_exp+1]+=exp
        aft_dt.init_dis+=dis
        
        aft_land=util.get_init_land(aft_dt,bldshps)
        print('dif land:',dis,np.linalg.norm(z.land-aft_land))
        
        image=cv2.imread(data[0].data[0].file_name+'jpg')
        
        for x,y,z in zip(z.land,aft_land,init_land):            
#            cv2.circle(aft_dt.img,tuple(np.around(x).astype(np.int)), 1, (0,255,0))
#            cv2.circle(aft_dt.img,tuple(np.around(y).astype(np.int)), 1, (0,0,255))
            cv2.circle(image,tuple(np.around(x).astype(np.int)), 2, (0,255,0), -1)
            cv2.circle(image,tuple(np.around(y).astype(np.int)), 2, (0,0,255), -1)
            cv2.circle(image,tuple(np.around(z).astype(np.int)), 1, (255,0,0), -1)
            
        cv2.imshow('test image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('D')
    
def test_set_img_stc(data,net):
    
    def gen_rand_data(data,all_data):
        for i in range(3):        
            data.angle[i]+=2*train.rand_angle_range[i]*np.random.random((1,))-train.rand_angle_range[i]
            
        for i in range(2):
            data.tslt[i]\
                +=2*train.rand_tslt_range[i]*np.random.random((1,))-train.rand_tslt_range[i]         
        data.tslt[2]\
            +=2*train.rand_tslt_range[2]*data.tslt[2]*np.random.random((1,))-train.rand_tslt_range[2]*data.tslt[2]
        
        for i in range(data.exp.shape[0]):
            if (i==0):
                continue
            data.exp[i]\
                +=2*train.rand_exp_range*np.random.random((1,))-train.rand_exp_range
# =============================================================================
#         first=random.randrange(len(all_data))
#         second=random.randrange(len(all_data[first].data))
#         data.exp[i]=all_data[first].data[second].exp.copy()
# =============================================================================
        
        first=random.randrange(len(all_data))
        second=random.randrange(len(all_data[first].data))
        data.dis=all_data[first].data[second].dis.copy()
            
    
    bldshps_path='/home/weiliu/fitting_dde/const_file/deal_data/blendshape_ide_svd_77.lv'
    bldshps=load.load_bldshps(bldshps_path,num_ide=77,num_exp=47,num_vtx=11510)
    util.bld_reduce_neu(bldshps)
    
    num_exp=bldshps.shape[1]-1
    num_land=data[0].data[0].land.shape[0]
    num_angle=3
    num_tslt=3
    #util.get_land(data[0].data[0],bldshps,data[0].user)
    
    ITE_num=1
    
    mean_ldmk,tri_idx,px_barycenter=load.load_tri_idx('../const_file/tri_idx_px.txt',num_land)
    
    max_cnt=100
    cnt=np.zeros((max_cnt),int)
    
    idx_n=0
    tot_num=0
    for one_ide in data:
        spf_bldshps=np.tensordot(bldshps,one_ide.user,axes=(0,0))
        
        for one_img in one_ide.data:
            p=np.random.random(1)[0]
            idx_n+=1
            print('image num now: %d tot image: %d' % (idx_n,tot_num))
            if (p>0.01):
                continue
            tot_num+=1             
            for ite in range(ITE_num):
                test_data=base.TestOnePoint(one_img,one_ide.user)
                std_land=test_data.land
                gen_rand_data(test_data,data)
#                bf_land=util.get_land(test_data,bldshps,one_ide.user)
                bf_land=util.get_land_spfbldshps(test_data,spf_bldshps)
                data_input=util.get_input_from_land_img(bf_land,test_data.img,tri_idx,px_barycenter)
                                
                out=net(torch.tensor(data_input.astype(np.float32)))
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
                    
    print(cnt)    
    pro=cnt.copy()    
    s=pro.sum()
    
    pro=pro/s
    for i in range(1,pro.shape[0]):
        pro[i]+=pro[i-1]
    print(pro)
    
def test_set_img_stc_dvd(data,net):
    
    def gen_rand_data(data,all_data,which):
        if which=='a':
            for i in range(3):        
                data.angle[i]+=2*train.rand_angle_range[i]*np.random.random((1,))-train.rand_angle_range[i]
        
        if which=='t':
        
        for i in range(2):
            data.tslt[i]\
                +=2*train.rand_tslt_range[i]*np.random.random((1,))-train.rand_tslt_range[i]         
        data.tslt[2]\
            +=2*train.rand_tslt_range[2]*data.tslt[2]*np.random.random((1,))-train.rand_tslt_range[2]*data.tslt[2]
        
        for i in range(data.exp.shape[0]):
            if (i==0):
                continue
            data.exp[i]\
                +=2*train.rand_exp_range*np.random.random((1,))-train.rand_exp_range
# =============================================================================
#         first=random.randrange(len(all_data))
#         second=random.randrange(len(all_data[first].data))
#         data.exp[i]=all_data[first].data[second].exp.copy()
# =============================================================================
        
        first=random.randrange(len(all_data))
        second=random.randrange(len(all_data[first].data))
        data.dis=all_data[first].data[second].dis.copy()
            
    
    bldshps_path='/home/weiliu/fitting_dde/const_file/deal_data/blendshape_ide_svd_77.lv'
    bldshps=load.load_bldshps(bldshps_path,num_ide=77,num_exp=47,num_vtx=11510)
    util.bld_reduce_neu(bldshps)
    
    num_exp=bldshps.shape[1]-1
    num_land=data[0].data[0].land.shape[0]
    num_angle=3
    num_tslt=3
    #util.get_land(data[0].data[0],bldshps,data[0].user)
    
    ITE_num=1
    
    mean_ldmk,tri_idx,px_barycenter=load.load_tri_idx('../const_file/tri_idx_px.txt',num_land)
    
    max_cnt=100
    cnt=np.zeros((max_cnt),int)
    
    idx_n=0
    tot_num=0
    for one_ide in data:
        spf_bldshps=np.tensordot(bldshps,one_ide.user,axes=(0,0))
        
        for one_img in one_ide.data:
            p=np.random.random(1)[0]
            idx_n+=1
            print('image num now: %d tot image: %d' % (idx_n,tot_num))
            if (p>0.01):
                continue
            tot_num+=1             
            for ite in range(ITE_num):
                test_data=base.TestOnePoint(one_img,one_ide.user)
                std_land=test_data.land
                gen_rand_data(test_data,data)
#                bf_land=util.get_land(test_data,bldshps,one_ide.user)
                bf_land=util.get_land_spfbldshps(test_data,spf_bldshps)
                data_input=util.get_input_from_land_img(bf_land,test_data.img,tri_idx,px_barycenter)
                                
                out=net(torch.tensor(data_input.astype(np.float32)))
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
                    
    print(cnt)    
    pro=cnt.copy()    
    s=pro.sum()
    
    pro=pro/s
    for i in range(1,pro.shape[0]):
        pro[i]+=pro[i-1]
    print(pro)
    
    
    
def load_one_data(path):
    
    data=[]
    print('load_one_data')
    load.load(data,path,'jpg',num_ide=77,num_exp=47,num_land=73)
    print(data)
    return data
    
def load_set_data():
    data=[]
    fwhs_path='/home/weiliu/fitting_dde/fitting_psp_f_l12_slt/fw'
    lfw_path='/home/weiliu/fitting_dde/fitting_psp_f_l12_slt/lfw_image'
    gtav_path='/home/weiliu/fitting_dde/fitting_psp_f_l12_slt/GTAV_image'
    
    
    load.load(data,fwhs_path,'jpg',num_ide=77,num_exp=47,num_land=73)
    load.load(data,lfw_path,'jpg',num_ide=77,num_exp=47,num_land=73)     
    load.load(data,gtav_path,'png',num_ide=77,num_exp=47,num_land=73)     
    
#    test_path='/home/weiliu/fitting_dde/4_psp_f_cal_test/data_me/test_only_three/'
#    load.load(data,test_path,'jpg',num_ide=77,num_exp=47,num_land=73)
    return data

    
net_model_path='../../model/net'+model_suffix+'.pkl'
net=torch.load(net_model_path,map_location='cpu')
# =============================================================================
# data=load_one_data('../data_test_one')
# test_set_img_show(data,net)
# =============================================================================
data=load_set_data()
test_set_img_stc(data,net)

