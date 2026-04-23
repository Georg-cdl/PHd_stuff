#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:59:07 2020

@author: tobias
"""
import cv2
import numpy as np

def rotatePoint(point,deg,center):


    M=cv2.getRotationMatrix2D(center,deg,1)



    t = np.hstack((np.array(point),1))


    
    result=np.round(np.matmul(M,t.transpose()).transpose())

    
    return result