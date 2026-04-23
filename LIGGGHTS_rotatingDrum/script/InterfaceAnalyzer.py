#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:25:48 2021

@author: tobias
"""

import argparse
import numpy as np
import struct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
import os
import pickle
import glob

import script.logging



log = script.logging.getLogger()


def getDataListFromPrev(file):
    if os.path.exists(file):   
        with open(file,"rb") as file:
            saveObj = pickle.load(file)
            print("State restored from file")
            (OFFSET_ANALYZE, already_analyzed, tracked_objs,overall_objs,last_cnt_list,last_cnt_list_pre,last_cnt_list_trans,last_cnt_list_pre_trans,objcount,vel_shape,data_list,area_list) = saveObj
            print('already_analyzed: '+str(already_analyzed))
            return data_list,area_list


def analyzeInterface(data_list,area_list):
    log.info("Interface analysis started")
    
    img = np.zeros((800,800),dtype=np.uint8)


    l=0
    for data in data_list:
        l=l+len(data[1])

    x_data =  np.zeros((l))
    y_data = np.zeros((l))
    index=0
    for data in data_list:
        length=len(data[1])
        x_data[index:index+length] = data[1]
        y_data[index:index+length] = data[0]
        index=index+length
        #plt.plot(data[1],data[0],'o')
    log.debug("start fit")
    fit1 = np.polyfit(x_data,y_data,1)
    log.debug("end fit")

    #plt.plot([0,img.shape[1]],[fit1[1],fit1[1]+fit1[0]*img.shape[1]])

    alpha = np.arctan(fit1[0])

    T=np.array([[np.cos(alpha), np.sin(alpha)],[-np.sin(alpha),np.cos(alpha)]])

    x=[0,img.shape[1]]
    y=[fit1[1],fit1[1]+fit1[0]*img.shape[1]]

    Test = T.dot(np.array([x_data,y_data]))
    
    
    s=50 # windowsize
    N=100
    x_min = 0
    x_max= 1200
    
    x_pos = np.linspace(0+s/2,1200,N)
    y_mean = np.zeros(N-1)
    y_std = np.zeros(N-1)
    
    N_points_window = np.zeros(N-1)
    
    x_step = (x_max-x_min-s)/(N-2)
    
    
    for i in range(N-1):
        xmin = x_min+i*x_step
        xmax = x_min+i*x_step+s
        m=np.logical_and(Test[0]>=xmin,Test[0]<xmax)
        j = Test[1][m]
        N_points_window[i] = len(j)
    
    
    N_points_window[N_points_window == 0] = np.nan
    N_points_window_mean = np.nanmean(N_points_window)
    log.debug("Mean count of points in window: {N}".format(N=np.round(N_points_window_mean)))

    for i in range(N-1):
        xmin = x_min+i*x_step
        xmax = x_min+i*x_step+s
        
        
        
        m=np.logical_and(Test[0]>=xmin,Test[0]<xmax)
        k = Test[0][m]
        j = Test[1][m]
        #print(len(j))
        if len(j)>N_points_window_mean*0.5 :
            y_mean[i] = np.mean(j) 
            y_std[i] = np.std(j)
        else:
            y_mean[i] = np.nan
            y_std[i] = np.nan
            
    
    #plt.figure()
    #plt.plot(x_pos[0:-1],y_mean)
    T2=np.array([[np.cos(alpha), -np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])
    Test2 = T2.dot(np.array([x_pos[0:-1],y_mean]))
    
    Test3a = T2.dot(np.array([x_pos[0:-1],y_mean+y_std]))
    Test3b = T2.dot(np.array([x_pos[0:-1],y_mean-y_std]))
    
    #plt.figure()
    
    #for data in data_list:
        #x_data = np.append(data[1],x_data)
        #y_data = np.append(data[0],y_data)

    plt.figure(figsize=(8,8))
    cv2.circle(img,(400,400), 400, 128, -1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.plot(Test2[0],Test2[1],'-ko')
    plt.plot(Test3a[0],Test3a[1],'-k')
    plt.plot(Test3b[0],Test3b[1],'-k')
    #plt.plot([0,img.shape[1]],[fit1[1],fit1[1]+fit1[0]*img.shape[1]])
    plt.savefig('analysis/contourImage.png')
    interface_ang = alpha/np.pi*180
    interface_dev = np.nanmean(y_std)/img.shape[1]*103
    log.info('interface angle: '+str(interface_ang))
    log.info('interface deviation: '+str(interface_dev))
    radius=400*0.95
    plt.figure()
    plt.plot(np.array(area_list)/(radius*radius*np.pi));plt.ylim([0,1]);
    log.info('area ratio: '+str(np.mean(np.array(area_list)/(radius*radius*np.pi))))
    plt.grid(True)
    plt.savefig('analysis/areaRatioDiagram.png')
    saveObj={'meanCont':Test2,'upperCont':Test3a,'lowerCont':Test3b,'interface_ang':interface_ang,'interface_dev':interface_dev,'areaRatioList':np.array(area_list)/(radius*radius*np.pi)}
    
    log.info("write cont file")
    with open('analysis/contData.pickle',"wb") as file:
            pickle.dump(saveObj,file)
            
    return {'IA':interface_ang,'ID':interface_dev, 'CM':{'x':list(Test2[0]),'y':list(Test2[1])},
            'CT':{'x':list(Test3a[0]),'y':list(Test3a[1])},'CL':{'x':list(Test3b[0]),'y':list(Test3b[1])}}
