#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 11:58:13 2021

@author: tobias
"""

import numpy as np

def getBlockCount(data):
    overall_objs, u, infos = data
         
    count=0
    for obj in overall_objs:
        count+=len(np.array(obj[3])[:,3])
    #print("block count {}".format(count))
    return count
    
def getIndividualBlockCount(data):
    overall_objs, u, infos = data
    count=len(overall_objs)
    #print("individual block count {}".format(count))
    return count

def getDistrubution(data,D,bins=40,norm=True,weight=False, h90=False):
    overall_objs, u, infos = data
    scaling = D/u[0]
    areas=[]
    block_heights_list2=[]

    for obj in overall_objs:
        if h90:
            block_heights_list2.append(np.array(obj[3],dtype=float)[:,2]*scaling)
            #print(np.array(obj[3])[:,2]*scaling)
        else:
            block_heights_list2.append(np.array(obj[3])[:,1]*scaling)
            #print(np.array(obj[3])[:,1]*scaling)
        areas.append(np.array(obj[3])[:,3])
    #x=np.concatenate(np.array(block_heights_list2))
    x = np.array([ elem for singleList in block_heights_list2 for elem in singleList])
    areas = np.array([ elem for singleList in areas for elem in singleList])
    #areas=np.concatenate(np.array(areas))
    if(weight):
        n,bins=np.histogram(x,weights=np.array(areas),bins=bins,range=(0,60),density=norm)
    else:
       n,bins=np.histogram(x,bins=bins,range=(0,60),density=norm)
    return bins,n

def getDistrubutionWr(data,D,bins=60,norm=True,weight=False, h90=False):
    overall_objs, u, infos = data
    scaling = D/u[0]
    areas=[]
    block_heights_list2=[]

    for obj in overall_objs:
        if h90:
            block_heights_list2.append(np.array(obj[3],dtype=float)[:,4]*scaling)
            #print(np.array(obj[3])[:,2]*scaling)
        else:
            block_heights_list2.append(np.array(obj[3])[:,0]*scaling)
            #print(np.array(obj[3])[:,1]*scaling)
        areas.append(np.array(obj[3])[:,3])
    #x=np.concatenate(np.array(block_heights_list2))
    x = np.array([ elem for singleList in block_heights_list2 for elem in singleList])
    areas = np.array([ elem for singleList in areas for elem in singleList])
    #areas=np.concatenate(np.array(areas))
    if(weight):
        n,bins=np.histogram(x,weights=np.array(areas),bins=bins,range=(0,103),density=norm)
    else:
       n,bins=np.histogram(x,bins=bins,range=(0,103),density=norm)
    return bins,n

def getAngleDistrubution(data,D,bins=40,norm=True,weight=False):
    overall_objs, u, infos = data
    areas=[]
    block_heights_list2=[]



    for obj in overall_objs:
        block_heights_list2.append(np.array(obj[5])[:,3])
        areas.append(np.array(obj[3])[:,3])
    x = np.array([ elem for singleList in block_heights_list2 for elem in singleList])
    #x=np.concatenate(np.array(block_heights_list2))
    areas = np.array([ elem for singleList in areas for elem in singleList])
    #areas=np.concatenate(np.array(areas))
    if(weight):
        n,bins=np.histogram(x,weights=np.array(areas),bins=bins,range=(0,90),density=norm)
    else:
       n,bins=np.histogram(x,bins=bins,range=(0,90),density=norm)

    return bins,n,(np.mean(x),np.average(x,weights=np.array(areas)))

def getTauDistrubution(data,D,bins=40,norm=True,weight=False):
    overall_objs, u, infos = data
    areas=[]
    block_heights_list2=[]
    scaling = D/u[0]
    for obj in overall_objs:
        
        block_heights_list2.append(np.array(obj[3],dtype=float)[:,2]*scaling*np.sin(np.array(obj[5])[:,3]/90*np.pi))
        #print(block_heights_list2[-1])
        areas.append(np.array(obj[3])[:,3])
    x = np.array([ elem for singleList in block_heights_list2 for elem in singleList])
    #x=np.concatenate(np.array(block_heights_list2))
    areas = np.array([ elem for singleList in areas for elem in singleList])
    #areas=np.concatenate(np.array(areas))
    if(weight):
        n,bins=np.histogram(x,weights=np.array(areas),range=(0,60),bins=bins,density=norm)
    else:
       n,bins=np.histogram(x,bins=bins,range=(0,90),density=norm)

    return bins,n,(np.mean(x),np.average(x,weights=np.array(areas)))

def getRotDistrubution(data,D,bins=40,norm=True,weight=False):
    overall_objs, u, infos = data

    fps=96
    scaling = D/u[0]
    #scaling3 = fps

    areas=[]
    block_heights_list2=[]

    for obj in overall_objs:
        block_heights_list2.append(np.array(obj[5])[:,4]*fps)
        areas.append(np.array(obj[3])[:,3])
    x =- np.array([ elem for singleList in block_heights_list2 for elem in singleList])
    #x=-np.concatenate(np.array(block_heights_list2))
    areas = np.array([ elem for singleList in areas for elem in singleList])
    #areas=np.concatenate(np.array(areas))
    if(weight):
        n,bins=np.histogram(x,weights=np.array(areas),bins=bins,range=(-5,15),density=norm)
    else:
       n,bins=np.histogram(x,bins=bins,range=(0,90),density=norm)

    return bins,n,(np.mean(x),np.average(x,weights=np.array(areas)))

def getGyrDistrubution(data,D,bins=50,norm=True,weight=False):
    overall_objs, u, infos = data
    areas=[]
    list2=[]

    for obj in overall_objs:
        list2.append(np.array(obj[5])[:,5])
        areas.append(np.array(obj[3])[:,3])
    x =- np.array([ elem for singleList in list2 for elem in singleList])
    areas = np.array([ elem for singleList in areas for elem in singleList])
    if(weight):
        n,bins=np.histogram(x,weights=np.array(areas),bins=bins,range=(-5,20),density=norm)
    else:
       n,bins=np.histogram(x,bins=bins,range=(-5,20),density=norm)

    return bins,n,(np.mean(x),np.average(x,weights=np.array(areas)))

def getSpeedDistrubution(data,D,bins=40,norm=True,weight=False):
    overall_objs, u, infos = data
    #print('size:' , u[0])
    fps=96
    scaling = D/u[0]
    scaling2 = D/u[0]*fps
    areas=[]
    block_heights_list2=[]

    for obj in overall_objs:
        block_heights_list2.append(np.array(obj[5])[:,0]*scaling2)
        areas.append(np.array(obj[3])[:,3])

    #x=np.concatenate(np.array(block_heights_list2))
    x=np.array([ elem for singleList in block_heights_list2 for elem in singleList])
    areas = np.array([ elem for singleList in areas for elem in singleList])
    #areas=np.concatenate(np.array(areas))
    if(weight):
        n,bins=np.histogram(x,weights=np.array(areas),bins=bins,range=(0,900),density=norm)
    else:
       n,bins=np.histogram(x,bins=bins,range=(0,90),density=norm)

    return bins,n,(np.mean(x),np.average(x,weights=np.array(areas)))

def getDistrubutionHUE(data,D,bins=40,norm=True,weight=False):
    overall_objs, u, infos = data
    scaling = D/u[0]
    areas=[]
    block_heights_list2=[]

    for obj in overall_objs:
        block_heights_list2.append((u[0]-np.array(obj[3],dtype=float)[:,5])*scaling)
        areas.append(np.array(obj[3])[:,3])
    #x=np.concatenate(np.array(block_heights_list2))
    x = np.array([ elem for singleList in block_heights_list2 for elem in singleList])
    areas = np.array([ elem for singleList in areas for elem in singleList])
    #areas=np.concatenate(np.array(areas))
    n,bins=np.histogram(x,weights=np.array(areas),bins=bins,range=(0,D),density=norm)
    return bins,n

def getDistrubutionAR(data,D,bins=40,norm=True,weight=False):
    overall_objs, u, infos = data


    x = infos['area_ratio']
    #print(np.mean(x))

    n,bins=np.histogram(x,bins=bins,range=(0.4,0.8),density=norm)
    return bins,n

def getMeanAR(data):
    overall_objs, u, infos = data


    x = infos['area_ratio']
    #print(np.mean(x))

    
    return np.mean(x)
