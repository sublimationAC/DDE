# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:49:14 2019

@author: Pavilion
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 11:22:44 2019

@author: Pavilion
"""

import numpy as np
import torch

import cv2
import sys
import copy
#import random
#import math

sys.path.append('../')

from load import load
from util import util
from base import base
 
model_suffix='_7_18_smpcnn_1'
    
def load_set_data():
    data=[]
    
    fwhs_path='/home/weiliu/fitting_dde/fitting_psp_f_l12_slt/fw'
    load.load(data,fwhs_path,'jpg',num_ide=77,num_exp=47,num_land=73)
    
    lfw_path='/home/weiliu/fitting_dde/fitting_psp_f_l12_slt/lfw_image'
    load.load(data,lfw_path,'jpg',num_ide=77,num_exp=47,num_land=73)     
    
    gtav_path='/home/weiliu/fitting_dde/fitting_psp_f_l12_slt/GTAV_image'
    load.load(data,gtav_path,'bmp',num_ide=77,num_exp=47,num_land=73)     
    
#    test_path='/home/weiliu/fitting_dde/4_psp_f_cal_test/data_me/test_only_three/'
#    load.load(data,test_path,'jpg',num_ide=77,num_exp=47,num_land=73)
    return data


            
            

def test_video_cnn(net,path):
    
    
#    bldshps_path='/home/weiliu/fitting_dde/const_file/deal_data/blendshape_ide_svd_77.lv'
    bldshps_path='/data/weiliu/fitting_psp_f_l12_slt/const_file/blendshape_ide_svd_77.lv'
    bldshps=load.load_bldshps(bldshps_path,num_ide=77,num_exp=47,num_vtx=11510)
    util.bld_reduce_neu(bldshps)
    
    num_exp=bldshps.shape[1]-1
    num_land=73
    num_angle=3
    num_tslt=3
    num_ide=bldshps.shape[0]
    
    
    test_data=base.DataCNNOne()
    user=np.empty((num_ide),dtype=np.float64)
    print(path+'_pre.psp_f')
    load.load_psp_f(path+'_pre.psp_f',test_data,user,num_ide,47,num_land)
    
#    load.load_img(path+'_first_frame.jpg',test_data)
#    load.load_land73(path+'_first_frame.land73',test_data,num_land)
    

    spf_bldshps=np.tensordot(bldshps,user,axes=(0,0))
    test_image=cv2.imread(path+'_first_frame.jpg')
    

        
    ip_vd = cv2.VideoCapture(path+'.mp4')
    
    fps = ip_vd.get(cv2.CAP_PROP_FPS)
    size = (int(ip_vd.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(ip_vd.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    ld_wt_vd = cv2.VideoWriter(path+model_suffix+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)    
    
    test_data.land=test_data.land_inv.copy()
    test_data.land[:,1]=size[1]-test_data.land_inv[:,1]
    test_data.centroid_inv=test_data.land_inv.mean(0)
    

    
    ini_tt_dt=copy.deepcopy(test_data)
    
    
    for idx in range(10000):
        ret, rgb_frame=ip_vd.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

    
        input_img=cv2.imread(path+'_230/lv_out_'+str(idx+1)+'.jpg') 
        ipt=util.norm_img(input_img).transpose(2,0,1)
        ipt=ipt[np.newaxis,:]
        result=np.squeeze(net(torch.tensor(ipt)).data.numpy())

        print(result.shape)
        angle=result[0:num_angle]
        tslt=result[num_angle:num_angle+num_tslt]
        exp=result[num_angle+num_tslt:num_angle+num_tslt+num_exp]
        dis=np.empty((num_land,2))
        dis[:,0]=result[num_angle+num_tslt+num_exp:num_angle+num_tslt+num_exp+num_land]
        dis[:,1]=result[num_angle+num_tslt+num_exp+num_land:num_angle+num_tslt+num_exp+num_land*2]

        test_data.angle=angle
        test_data.tslt=tslt
        test_data.exp[1:num_exp+1]=exp
        test_data.dis=dis
        
        
        test_data.land_inv=util.get_land_spfbldshps_inv(test_data,spf_bldshps)
        test_data.land=test_data.land_inv.copy()
        test_data.land[:,1]=gray_frame.shape[0]-test_data.land_inv[:,1]
        
        test_data.centroid_inv=test_data.land_inv.mean(0)
        
        print('dif vs ini:',ini_tt_dt.angle-test_data.angle)
        print('dif vs ini:',ini_tt_dt.tslt-test_data.tslt)
        
        for pt in test_data.land:
            cv2.circle(rgb_frame,tuple(np.around(pt).astype(np.int)), 2, (0,255,0), -1)
        
        ld_wt_vd.write(rgb_frame)
        
        load.save_psp_f(path+'_psp_f/lv_out_psp_f_tt_'+str(idx)+'.psp_f',test_data,user)
        
    '''          
        cv2.imshow('test frame',rgb_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''
    ip_vd.release()
    ld_wt_vd.release()


net_model_path='../../model/net'+model_suffix+'.pkl'
net=torch.load(net_model_path,map_location='cpu')
#data=load_set_data()
test_video_cnn(net,'../video_data/lv_out')

