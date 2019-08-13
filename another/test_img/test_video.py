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
 

model_suffix='_7_18_cf3_norm_gpu_lr5_1w_bch1w_hid2'
    
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



def test_video(train_data,net,path):
    
    def update_train_data(train_data,spf_bldshps, f_vd,center_vd, tslt2):
        test_all_data=[]
        for one_ide in train_data:
            for one_img in one_ide.data:
                test_all_data.append(base.TestVideoOnePoint(one_img))

                test_all_data[-1].fcs=f_vd
                test_all_data[-1].center=center_vd
                test_all_data[-1].tslt[2]=tslt2
                
                test_all_data[-1].land_inv=util.get_land_spfbldshps_inv(test_all_data[-1],spf_bldshps)
#                test_all_data[-1].land=test_all_data[-1].land_inv.copy()
#                test_all_data[-1].land[:,1]=one_img.img.shape[0]-test_all_data[-1].land_inv[:,1]
                
                test_all_data[-1].centroid_inv=test_all_data[-1].land_inv.mean(0)
        
        return test_all_data
    
    def get_init_shape(init_num,all_test_data,test_data,spf_bldshps):
        ans=[0]*init_num
        ans_dis=[1e9]*init_num
        for i in range(init_num):
            ans[i]=all_test_data[i]
        
        for one_shape in all_test_data:
            dis=np.linalg.norm((one_shape.land_inv-test_data.land_inv)-
                               (one_shape.centroid_inv-test_data.centroid_inv))
            for i in range(init_num):
                if (ans_dis[i]>dis):
                    ans[i+1:]=ans[i:-1]
                    ans_dis[i+1:]=ans_dis[i:-1]
                    ans[i]=one_shape
                    ans_dis[i]=dis
                    break
            #tslt?

            
#        return ans 
#        print(ans[:].tslt)
#        print(test_data.tslt)
        print(ans_dis)
        ansx=copy.deepcopy(ans)
        ans_dis_aft=ans_dis.copy()
        for x,y in zip(ansx,ans_dis_aft):
            x.tslt[:2]+=(-x.centroid_inv+test_data.centroid_inv)/test_data.fcs*x.tslt[2]
            util.get_slt_land_cor(x,spf_bldshps,x.exp)
            land_inv=util.get_land_spfbldshps_inv(x,spf_bldshps)
            y=np.linalg.norm((land_inv-test_data.land_inv))
            
        print(ansx[0].tslt)
        print(ans[0].tslt)
        print(test_data.tslt)
        
        
        return ansx,ans_dis,ans_dis_aft
    
    def get_init_shape_inner(init_num,all_test_data,test_data,spf_bldshps):
        ans=[0]*init_num
        ans_dis=[1e9]*init_num
        for i in range(init_num):
            ans[i]=all_test_data[i]
        
        for one_shape in all_test_data:
            dis=np.linalg.norm((one_shape.land_inv[15:]-test_data.land_inv[15:])-
                               (one_shape.centroid_inv-test_data.centroid_inv))
            for i in range(init_num):
                if (ans_dis[i]>dis):
                    ans[i+1:]=ans[i:-1]
                    ans_dis[i+1:]=ans_dis[i:-1]
                    ans[i]=one_shape
                    ans_dis[i]=dis
                    break
            #tslt?

            
#        return ans 
#        print(ans[:].tslt)
#        print(test_data.tslt)
        print(ans_dis)
        ansx=copy.deepcopy(ans)
        ans_dis_aft=ans_dis.copy()
        for x,y in zip(ansx,ans_dis_aft):
            x.tslt[:2]+=(-x.centroid_inv+test_data.centroid_inv)/test_data.fcs*x.tslt[2]
            util.get_slt_land_cor(x,spf_bldshps,x.exp)
            land_inv=util.get_land_spfbldshps_inv(x,spf_bldshps)
            y=np.linalg.norm((land_inv[15:]-test_data.land_inv[15:]))
            
        print(ansx[0].tslt)
        print(ans[0].tslt)
        print(test_data.tslt)
        
        
        return ansx,ans_dis,ans_dis_aft
    
    bldshps_path='/home/weiliu/fitting_dde/const_file/deal_data/blendshape_ide_svd_77.lv'
    bldshps=load.load_bldshps(bldshps_path,num_ide=77,num_exp=47,num_vtx=11510)
    util.bld_reduce_neu(bldshps)
    
    num_exp=bldshps.shape[1]-1
    num_land=train_data[0].data[0].land.shape[0]
    num_angle=3
    num_tslt=3
    num_ide=bldshps.shape[0]
    
    
    test_data=base.TestVideoOnePoint(train_data[0].data[0])
    user=np.empty((num_ide),dtype=np.float64)
    print(path+'_pre.psp_f')
    load.load_psp_f(path+'_pre.psp_f',test_data,user,num_ide,47,num_land)
    
#    load.load_img(path+'_first_frame.jpg',test_data)
#    load.load_land73(path+'_first_frame.land73',test_data,num_land)
    

    spf_bldshps=np.tensordot(bldshps,user,axes=(0,0))
    test_image=cv2.imread(path+'_first_frame.jpg')
    
    all_test_data=update_train_data(train_data,spf_bldshps,test_data.fcs,test_data.center,test_data.tslt[2])
    
#    for x in all_test_data:
#        for pt in x.land_inv:
#            cv2.circle(test_image,tuple(np.around(pt).astype(np.int)), 1, (255,255,255), -1)
#    cv2.imshow('test frame',test_image)
#    cv2.waitKey(0)        
    
    
    mean_ldmk,tri_idx,px_barycenter=load.load_tri_idx('../const_file/tri_idx_px.txt',num_land)
    
    ip_vd = cv2.VideoCapture(path+'.mp4')
    
    fps = ip_vd.get(cv2.CAP_PROP_FPS)
    size = (int(ip_vd.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(ip_vd.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    init_num=1
    ld_wt_vd = cv2.VideoWriter(path+model_suffix+'_initshape'+str(init_num)+'_inner.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)    
    
    test_data.land=test_data.land_inv.copy()
    test_data.land[:,1]=size[1]-test_data.land_inv[:,1]
    test_data.centroid_inv=test_data.land_inv.mean(0)    
    
    ini_tt_dt=copy.deepcopy(test_data)
    init_land_data=copy.deepcopy(test_data)
    
    dbg_init_dis_file=open('dbg_init_dis_file.txt','w')
    dbg_dis_norm_file=open('dis_norm_file.txt','w')
#    dbg_init_aft_dis_file=open('../mid_data/dbg_init_aft_dis_file.txt','w')
    
    
    for idx in range(10000):
        ret, rgb_frame=ip_vd.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

        for pt in test_data.land:
            cv2.circle(rgb_frame,tuple(np.around(pt).astype(np.int)), 2, (255,0,0), -1)
        
        init_shape,dis,ans_dis_aft=get_init_shape_inner(init_num,all_test_data,test_data,spf_bldshps)
        
        dbg_init_dis_file.write(str([idx,dis,ans_dis_aft])+'\n')
        
        
#        result=np.zeros((num_angle+num_tslt+num_exp+2*num_land,),dtype=np.float64)
        angle,tslt=np.zeros(num_angle,dtype=np.float64),np.zeros(num_tslt,dtype=np.float64)
        exp,dis=np.zeros(num_exp,dtype=np.float64),np.zeros((num_land,2),dtype=np.float64)
        
        for x in init_shape:
            land=util.get_land_spfbldshps_inv(x,spf_bldshps)
            for pt in land:
                cv2.circle(rgb_frame,tuple(np.around(pt).astype(np.int)), 1, (122,122,122), -1)
            land[:,1]=gray_frame.shape[0]-land[:,1]
            for pt in land:
                cv2.circle(rgb_frame,tuple(np.around(pt).astype(np.int)), 2, (255,255,255), -1)
            input_now=util.get_input_from_land_img(land,gray_frame,tri_idx,px_barycenter)
            print(input_now.shape)
            result=net(torch.tensor(input_now.astype(np.float32))).data.numpy()
            
            print(result)
            print('f: ', test_data.fcs, x.fcs)
            print(test_data.angle,test_data.tslt)
            print(x.angle,x.tslt)
            print(result[0:num_angle],result[num_angle:num_angle+num_tslt])
            print(x.angle+result[0:num_angle],x.tslt+result[num_angle:num_angle+num_tslt])
            print('dis x:',x.dis[:,0]+result[num_angle+num_tslt+num_exp:num_angle+num_tslt+num_exp+num_land])
            print('dis y:',x.dis[:,0]+result[num_angle+num_tslt+num_exp+num_land:num_angle+num_tslt+num_exp+num_land*2])
            angle+=x.angle+result[0:num_angle]
            tslt+=x.tslt+result[num_angle:num_angle+num_tslt]
            exp+=x.exp[1:]+result[num_angle+num_tslt:num_angle+num_tslt+num_exp]
            dis[:,0]+=x.dis[:,0]+result[num_angle+num_tslt+num_exp:num_angle+num_tslt+num_exp+num_land]
            dis[:,1]+=x.dis[:,1]+result[num_angle+num_tslt+num_exp+num_land:num_angle+num_tslt+num_exp+num_land*2]
            
            init_land_data.angle=x.angle+result[0:num_angle]
            init_land_data.tslt=x.tslt+result[num_angle:num_angle+num_tslt]
            init_land_data.exp[1:]=x.exp[1:]+result[num_angle+num_tslt:num_angle+num_tslt+num_exp]
            init_land_data.dis[:,0]=x.dis[:,0]+result[num_angle+num_tslt+num_exp:num_angle+num_tslt+num_exp+num_land]
            init_land_data.dis[:,0]=x.dis[:,1]+result[num_angle+num_tslt+num_exp+num_land:num_angle+num_tslt+num_exp+num_land*2]
            
            init_shape_land=util.get_land_spfbldshps_inv(init_land_data,spf_bldshps)
            init_shape_land[:,1]=gray_frame.shape[0]-init_shape_land[:,1]
# =============================================================================
#             print(result)
#             for pt in init_shape_land:
#                 cv2.circle(rgb_frame,tuple(np.around(pt).astype(np.int)), 1, (0,0,255), -1)
#             cv2.imshow('test frame',rgb_frame)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
# =============================================================================
    
        test_data.angle=angle/init_num
        test_data.tslt=tslt/init_num
        test_data.exp[1:num_exp+1]=exp/init_num
        test_data.dis=dis/init_num
        test_data.land_cor=init_shape[0].land_cor
        
        test_data.land_inv=util.get_land_spfbldshps_inv(test_data,spf_bldshps)
        test_data.land=test_data.land_inv.copy()
        test_data.land[:,1]=gray_frame.shape[0]-test_data.land_inv[:,1]
        
        test_data.centroid_inv=test_data.land_inv.mean(0)
        
        print('dif vs ini:',ini_tt_dt.angle-test_data.angle)
        print('dif vs ini:',ini_tt_dt.tslt-test_data.tslt)      
        
        dbg_dis_norm_file.write(str(np.linalg.norm(test_data.dis))+'\n')
        
        for pt in test_data.land:
            cv2.circle(rgb_frame,tuple(np.around(pt).astype(np.int)), 2, (0,255,0), -1)
        
        ld_wt_vd.write(rgb_frame)
        
        load.save_psp_f(path+'_psp_f/lv_out_psp_f_tt_'+str(idx)+'.psp_f',test_data,user)
        
# =============================================================================
#         if idx>90:            
#             cv2.imshow('test frame',rgb_frame)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
# =============================================================================
    
    
    ip_vd.release()
    ld_wt_vd.release()
    dbg_init_dis_file.close()
    dbg_dis_norm_file.close()

#net_model_path='../mid_data/net_7_10_cf1_norm_1w.pkl'
#net_model_path='../mid_data/net_7_8_cf1_norm_exp.pkl'
#net_model_path='../mid_data/net_7_10_cf2_norm_5k.pkl'
net_model_path='../../model/net'+model_suffix+'.pkl'
net=torch.load(net_model_path,map_location='cpu')
data=load_set_data()
test_video(data,net,'../video_data/lv_out/lv_out')
#test_video(data,net,'../video_data/lv_small/lv_small')
