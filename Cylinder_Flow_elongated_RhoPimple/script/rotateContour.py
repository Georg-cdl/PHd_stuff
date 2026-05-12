#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:56:05 2020

@author: tobias
"""

import cv2
import numpy as np

def rotateCont(cnt,deg):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    #center=(int(img.shape[0]/2),int(img.shape[1]/2))
    center = (cx,cy)
    M=cv2.getRotationMatrix2D(center,deg,1)


    cnt= np.squeeze(cnt)
    t = np.ones((cnt.shape[0],cnt.shape[1]+1)); t[:,:-1] = cnt


    result=np.zeros((cnt.shape[0],1,cnt.shape[1]),dtype=np.int32)
    result[:,0,:]=np.round(np.matmul(M,t.transpose()).transpose())

    
    return result