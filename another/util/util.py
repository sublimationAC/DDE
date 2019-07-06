# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:40:45 2019

@author: Pavilion
"""

import numpy as np
from math import cos, sin, acos, asin, fabs, sqrt

def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left. 
        z: roll. positive for tilting head right. 
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  -sin(x)],
                 [0, sin(x),   cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, sin(y)],
                 [      0, 1,      0],
                 [-sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), -sin(z), 0],
                 [sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    
    R=Rz.dot(Ry.dot(Rx))
    return R.astype(np.float32)

def angle2matrix_3ddfa(angles):
    ''' get rotation matrix from three rotation angles(radian). The same as in 3DDFA.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch.
        y: yaw. 
        z: roll. 
    Returns:
        R: 3x3. rotation matrix.
    '''
    # x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    x, y, z = angles[0], angles[1], angles[2]
    
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  sin(x)],
                 [0, -sin(x),   cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, -sin(y)],
                 [      0, 1,      0],
                 [sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), sin(z), 0],
                 [-sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    R = Rx.dot(Ry).dot(Rz)
    return R.astype(np.float32)

def angle2matrix_zyx(angles):

    x, y, z = angles[2], angles[1], angles[0]
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  -sin(x)],
                 [0, sin(x),   cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, sin(y)],
                 [      0, 1,      0],
                 [-sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), -sin(z), 0],
                 [sin(z),  cos(z), 0],
                 [     0,       0, 1]])
        
    R=Rx.dot(Ry.dot(Rz))
    return R.astype(np.float64)


def matrix2uler_angle_zyx(rot):
    x,y=rot[0],rot[1]
    
    assert((1-x[2]**2)>(1e-5))
    
    be=asin(max(-1,min(1,x[2])))
    al=asin(max(-1,min(1,-x[1]/sqrt(1-x[2]**2))))
    ga=asin(max(-1,min(1,-y[2]/sqrt(1-x[2]**2))))
    
    
    return np.array([al,be,ga])
    
def bld_reduce_neu(bldshps):
    
    for x in bldshps:
        t=x[0].copy()
        x-=t
        x[0]=t

    

def recal_dis(data,bldshps):
#    print(data.land_cor)
    ldmk_bld=bldshps[:,:,data.land_cor,:]
#    print(ldmk_bld.shape)
    landmk_3d=np.tensordot(np.tensordot(ldmk_bld,data.user,axes=(0,0)),data.exp,axes=(0,0))
    
    landmk_3d=(angle2matrix_zyx(data.angle)@landmk_3d.T).T+data.tslt
    
    landmk_3d[:,0]/=landmk_3d[:,2]
    landmk_3d[:,1]/=landmk_3d[:,2]
    
    landmk_3d*=data.fcs
    
#    print('center-------',data.center)
    landmk_3d[:,0:2]+=data.center
    
    data.dis=(data.land_inv-landmk_3d[:,0:2]).copy()
    
def get_init_land(data,bldshps):
    
    ldmk_bld=bldshps[:,:,data.land_cor,:]
    
    landmk_3d=np.tensordot(np.tensordot(ldmk_bld,data.user,axes=(0,0)),data.init_exp,axes=(0,0))
    
    landmk_3d=(angle2matrix_zyx(data.init_angle)@landmk_3d.T).T+data.init_tslt
    
    
    landmk_3d[:,0]/=landmk_3d[:,2]
    landmk_3d[:,1]/=landmk_3d[:,2]
    
    landmk_3d*=data.fcs
    
# =============================================================================
#     print('center-------',data.center)
#     print(landmk_3d[:10,0:2])
# =============================================================================
    landmk_3d[:,0:2]+=data.center
# =============================================================================
#     print(landmk_3d[:10,0:2])
#     print('###################################')
#     print(data.init_dis[:5])
# =============================================================================
    
    ans=(landmk_3d[:,0:2]+data.init_dis).copy()

    
    ans[:,1]=data.img.shape[0]-ans[:,1]
    
# =============================================================================
#     print(np.concatenate((ans, data.land), axis=1))    
#     print(np.concatenate((data.angle, data.init_angle), axis=0))   
#     print(np.concatenate((data.tslt, data.init_tslt), axis=0))   
#     print(data.exp)
#     print(data.init_exp)
# =============================================================================
    
    
    return ans
    
def get_land(data,bldshps,user):
    
    ldmk_bld=bldshps[:,:,data.land_cor,:]
    
    landmk_3d=np.tensordot(np.tensordot(ldmk_bld,user,axes=(0,0)),data.exp,axes=(0,0))
    
    landmk_3d=(angle2matrix_zyx(data.angle)@landmk_3d.T).T+data.tslt
    landmk_3d_=(data.rot@landmk_3d.T).T+data.tslt
    print(np.concatenate((landmk_3d, landmk_3d_), axis=1))
    
    
    landmk_3d[:,0]/=landmk_3d[:,2]
    landmk_3d[:,1]/=landmk_3d[:,2]
    
    landmk_3d*=data.fcs
    
    print('center-------',data.center)
    print(landmk_3d[:10,0:2])
    landmk_3d[:,0:2]+=data.center
    print(landmk_3d[:10,0:2])
    print('###################################')
    print(data.dis[:5])
    
    ans=(landmk_3d[:,0:2]+data.dis).copy()
        
    ans[:,1]=data.img.shape[0]-ans[:,1]
    
    print(np.concatenate((ans, data.land), axis=1))    

    
    return ans    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    