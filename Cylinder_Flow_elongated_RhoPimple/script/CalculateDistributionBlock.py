#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:37:31 2020

@author: tobias
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import script.logging

log = script.logging.getLogger()

from script.blockValueDistribution import getBlockCount, getIndividualBlockCount, \
getDistrubution, getDistrubutionWr, getAngleDistrubution, getTauDistrubution, \
getRotDistrubution, getSpeedDistrubution, getDistrubutionHUE, getDistrubutionAR, getGyrDistrubution

MIN_BLOCKS = 10
MIN_INDIVIDUAL_BLOCKS = 2
SCORE_INVALID = 100

def getBlockDataFromPickle(file):
    with open(file,"rb") as f:
        pd = pickle.load(f)
        if len(pd)==2:
            overall_objs,u = pd
            infos = None
        if len(pd)==3:
         	overall_objs,u, infos = pd
        if type(u)!=tuple:
        		u=u.shape
        return overall_objs, u, infos
    return None

def getDictDistribution(bins_list,n_list):
    bins = list(np.mean(np.array(bins_list),axis=0))
    n = list(np.mean(np.array(n_list),axis=0))
    cum = list(np.cumsum(np.mean(np.array(n_list),axis=0),axis=0)/np.sum(np.mean(np.array(n_list),axis=0)))
    return {"bins":bins, "n":n, "cum":cum}
    
def calculateDistributions(files):

    n=True
    weight=True
    h90=False
    bins_list=[]
    n_list=[]
    
    angle_bins_list=[]
    angle_n_list=[]
    angle_mean_list=[]
    
    rot_bins_list=[]
    rot_n_list=[]
    rot_mean_list=[]

    gyr_bins_list=[]
    gyr_n_list=[]
    gyr_mean_list=[]
    
    speed_bins_list=[]
    speed_n_list=[]
    speed_mean_list=[]
    
    bins_list_hue98=[]
    n_list_hue98=[]
    
    bins_list_h98=[]
    n_list_h98=[]
    
    bins_list_wr=[]
    n_list_wr=[]
    
    bins_list_ar=[]
    n_list_ar=[]
    
    for file in files:
        f = getBlockDataFromPickle(file)
        blockCount = getBlockCount(f)
        log.info("block count: {}".format(blockCount))
        if blockCount<MIN_BLOCKS:
            log.warning("block count to low for distributions")
            return -1
        
        individualBlockCount = getIndividualBlockCount(f)
        log.info("individual block count: {}".format(individualBlockCount))
        if individualBlockCount<MIN_INDIVIDUAL_BLOCKS:
            log.warning("individual block count to low for distributions")
            return -1
        
                
        bins, n ,means=getRotDistrubution(f,103,norm=True,weight=weight)
        rot_bins_list.append(bins)
        rot_n_list.append(n)
        rot_mean_list.append(means)
        
        bins, n ,means=getGyrDistrubution(f,103,norm=True,weight=weight)
        gyr_bins_list.append(bins)
        gyr_n_list.append(n)
        gyr_mean_list.append(means)
        
        bins, n ,means=getAngleDistrubution(f,103,norm=True,weight=weight)
        angle_bins_list.append(bins)
        angle_n_list.append(n)
        angle_mean_list.append(means)
        
        bins, n ,means=getSpeedDistrubution(f,103,norm=True,weight=weight)
        speed_bins_list.append(bins)
        speed_n_list.append(n)
        speed_mean_list.append(means)
        
        bins, n =getDistrubutionHUE(f,103,norm=True)
        bins_list_hue98.append(bins)
        n_list_hue98.append(n)
        
        bins, n =getDistrubution(f,103,norm=True,weight=True,h90=True)
        bins_list_h98.append(bins)
        n_list_h98.append(n)

        bins, n =getDistrubutionWr(f,103,norm=True,weight=True,h90=True)
        bins_list_wr.append(bins)
        n_list_wr.append(n)
        
        bins, n = getDistrubutionAR(f,103,norm=True,weight=True)
        bins_list_ar.append(bins)
        n_list_ar.append(n)
    
    result_dict = {}
    result_dict["BH"] = getDictDistribution(bins_list_h98,n_list_h98)
    result_dict["BW"] = getDictDistribution(bins_list_wr,n_list_wr)
    result_dict["BA"] = getDictDistribution(angle_bins_list,angle_n_list)
    result_dict["BS"] = getDictDistribution(speed_bins_list,speed_n_list)
    result_dict["BR"] = getDictDistribution(rot_bins_list,rot_n_list)
    result_dict["BG"] = getDictDistribution(gyr_bins_list,gyr_n_list)
    result_dict["BT"] = getDictDistribution(bins_list_hue98,n_list_hue98)
    result_dict["AR"] = getDictDistribution(bins_list_ar,n_list_ar)
    return result_dict
        

             
# =============================================================================
#   
# 
# def compareBlocksData(target,sample):
#     global name_list
#     global h98_list
#     global w98_list
#     global max_list
#     global ang_list
#     global rot_list
#     global speed_list
#     global hue98_list
#     global area_ratio_list
#     
#     global icount_list
#     global count_list
#     
#     area_ratio_list=[]
#     name_list=[]
#     h98_list=[]
#     w98_list=[]
#     max_list=[]
#     ang_list=[]
#     rot_list=[]
#     speed_list=[]
#     hue98_list = []
#     icount_list=[]
#     count_list=[]
#     #print(target)
#     files=glob.glob(target)
#     files.sort()
#     #print(files)
#     doPlots(files,prefix='Mo4.6 40%RH',meanCol='k',singleValues=False,showLegend=False,output="")
#     
#     
#     
#     files=glob.glob(sample)
#     print(files)
#     files = files[0:1]
#     dat=files[0].split("_")[-1].split(".")[0]
#     k=1
#     if doPlots(files,prefix='DEM sim '+dat,meanCol='C1',singleValues=False,showLegend=False,output="") != 0:
#         print('do plots failed')
#         score = SCORE_INVALID
#         detail = {'count_individual':icount_list[-1], 'count_all':count_list[-1]}
#     else:
#         k_h98 = 1/60
#         k_ang=1/90
#         k_rot = 1/50
#         k_speed = 1/800
#         k_w98 = 1/60
#         k_hue98 = 1/80
#         
#         k_area_ratio = 1E-10 #75
#         
#         
#         
#         print('area_ratio_list')
#         print(area_ratio_list)
#         
#         
#         values = np.array([k_h98*np.sum(np.abs(np.array(h98_list[k])-np.array(h98_list[0]))),
#                            k_speed*np.sum(np.abs(np.array(speed_list[k])-np.array(speed_list[0]))),
#                            k_ang*np.sum(np.abs(np.array(ang_list[k])-np.array(ang_list[0]))),
#                            k_rot*np.sum(np.abs(np.array(rot_list[k])-np.array(rot_list[0]))),
#                            k_w98*np.sum(np.abs(np.array(w98_list[k])-np.array(w98_list[0]))),
#                            k_hue98*np.sum(np.abs(np.array(hue98_list[k])-np.array(hue98_list[0]))),
#                            k_area_ratio*np.sum(np.abs(np.array(area_ratio_list[k])-np.array(area_ratio_list[0])))])
#         
#         
#     
#         score = np.sqrt(np.sum(values**2))
#         #detail = np.hstack(([icount_list[-1],count_list[-1]],values))
#         print(values)
#         
#         detail = {'count_individual':icount_list[-1], 'count_all':count_list[-1], 'score_h98':values[0], 'score_speed':values[1], 'score_angle':values[2], 'score_rot':values[3], 'score_w98':values[4], 'score_hue98':values[5], 'score_ar':values[6]}
#      
# 
#     
#     
#     #print(icount_list)
#     #print("HALLO")
#     return score, detail
# 
# 
# 
# =============================================================================
if __name__ == "__main__":
    

    target = "/home/tobias/workspace/analyzeRoutines/Blockanalyzer/PlotRoutines/2021_10_08_Reutte/Mo46_raw_comparison/in/0_Mo4.6/P1200040*blocks.pickle"
    dist = calculateDistributions(glob.glob(target))
    
    plt.plot(dist["AR"]["bins"][1:],dist["AR"]["cum"])


