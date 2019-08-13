# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:34:36 2019

@author: Pavilion
"""
import numpy as np
import random
import math
import sys
import cv2

sys.path.append('../')
from base import base
from load import load
from util import util
from net_regress import smp_cnn_net


def get_data_from_file():
    
    data=[]
    fwhs_path='/data/weiliu/fitting_psp_f_l12_slt/FaceWarehouse'
    lfw_path='/data/weiliu/fitting_psp_f_l12_slt/lfw_image'
    gtav_path='/data/weiliu/fitting_psp_f_l12_slt/GTAV_image'
    test_path='/data/weiliu/fitting_psp_f_l12_slt/test_only_one'
    
    load.load_cnn(data,fwhs_path,'jpg',num_ide=77,num_exp=47,num_land=73)
    load.load_cnn(data,lfw_path,'jpg',num_ide=77,num_exp=47,num_land=73)     
    load.load_cnn(data,gtav_path,'bmp',num_ide=77,num_exp=47,num_land=73)  
#    load.load_cnn(data,test_path,'jpg',num_ide=77,num_exp=47,num_land=73)  
    
    tt_num=0
    for one_ide in data:
        tt_num+=len(one_ide.data)
    
    print(tt_num)
    print(len(data[0].data))
    
    num_exp=data[0].data[0].exp.shape[0]-1
    num_land=data[0].data[0].land.shape[0]
    num_angle=3
    num_tslt=3    
    
#    cv2.imshow('test image',data[0].data[0].img_230)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    
    img_230_height=data[0].data[0].img_230.shape[0]
    img_230_width=data[0].data[0].img_230.shape[1]
    img_230_channel=data[0].data[0].img_230.shape[2]
    
    input_cnn_data=np.empty(
                    (tt_num,img_230_channel,img_230_height,img_230_width)
                    ,dtype=np.float32)
    output_cnn_data=np.empty((tt_num,num_angle+num_tslt+num_exp+num_land*2),dtype=np.float32)

    counter=0
    for one_ide in data:
        for one_data in one_ide.data:
            
            input_cnn_data[counter]=util.norm_img(one_data.img_230).transpose(2,0,1)

#            output_cnn_data[counter, 0:num_angle]\
#                =(one_data.angle-one_data.init_angle).copy()
#            output_cnn_data[counter,num_angle:num_angle+num_tslt]\
#                =(one_data.tslt-one_data.init_tslt).copy()
#            output_cnn_data[counter,num_angle+num_tslt:num_angle+num_tslt+num_exp]\
#                =(one_data.exp-one_data.init_exp)[1:num_exp+1].copy()
#            output_cnn_data[counter,num_angle+num_tslt+num_exp:num_angle+num_tslt+num_exp+num_land]\
#                =(one_data.dis-one_data.init_dis)[:,0].copy()
#            output_cnn_data[counter,num_angle+num_tslt+num_exp+num_land:num_angle+num_tslt+num_exp+num_land*2]\
#                =(one_data.dis-one_data.init_dis)[:,1].copy()
            output_cnn_data[counter, 0:num_angle]\
                =(one_data.angle).copy()
            output_cnn_data[counter,num_angle:num_angle+num_tslt]\
                =(one_data.tslt).copy()
            output_cnn_data[counter,num_angle+num_tslt:num_angle+num_tslt+num_exp]\
                =(one_data.exp)[1:num_exp+1].copy()
            output_cnn_data[counter,num_angle+num_tslt+num_exp:num_angle+num_tslt+num_exp+num_land]\
                =(one_data.dis)[:,0].copy()
            output_cnn_data[counter,num_angle+num_tslt+num_exp+num_land:num_angle+num_tslt+num_exp+num_land*2]\
                =(one_data.dis)[:,1].copy()


                
            counter+=1


    return input_cnn_data,output_cnn_data

ipt_data,opt_data=get_data_from_file()
smp_cnn_net.train_smp_cnn(ipt_data,opt_data)
                