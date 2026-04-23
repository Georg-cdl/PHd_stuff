#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:39:01 2020

@author: tobias
"""

import multiprocessing as mp
import os
import numpy as np
import math
import cv2
import vtk
from vtk.util.numpy_support import vtk_to_numpy
#from numba import jit
import glob

import time
import pickle

from script.findNonReflective import findNonreflectiveSimilarity

from script.rotateContour import rotateCont
from script.rotatePoint import rotatePoint

import script.CalculateDistributionBlock
import script.InterfaceAnalyzer
import script.logging

import argparse


import re

log = script.logging.getLogger()


OFFSET_ANALYZE = 540

FPS = 96

debug_write_images=False
debug_plotSizemeasure=False

datafolder = 'tmp'

def saveDrumImage(data,outfile):    
    img = 255*cv2.exp(-0.1*(data[2]-1))
    img = cv2.flip(img, 1)
    img= cv2.merge([img, img, img])
    src1 = np.zeros((800,800,3))+(255,255,255)
    src2= np.zeros((800,800,3))+(70,70,70)
    foreground = src1.astype(float)
    background = src2.astype(float)
    alpha = img.astype(float)/255
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    img = cv2.add(foreground, background)
    cv2.circle(img, (400,400), 400, (0,0,255), 2)
    cv2.imwrite(outfile, img)

def kernelprecompute(N):
    data=np.zeros((N,N,N),dtype=bool)    
    for i in range(2,N): 
        data[0:i,0:i,i] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(i,i))
    return data
#@jit(nopython=True)
def getkernel(data,i):
    return data[0:i,0:i,i]

def readVTKDrumFile(filename,resolution,drumDiameter,rpm,fps,displacement=True):
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    reader.ReadAllScalarsOn()  # Activate the reading of all scalars
    reader.Update()
    
    data=reader.GetOutput()
    pos = vtk_to_numpy(data.GetPoints().GetData())
    R = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray("radius"))
    v = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray("v"))
    
    reader.CloseVTKFile()
    del reader

    kernels = kernelprecompute(int(np.max(R)/drumDiameter*resolution)*2+1+1)
    img = renderData(pos,R,v,resolution,drumDiameter,rpm,fps,displacement,kernels)
    img = tuple(x.astype('float32') for x in img)
    saveDrumImage(img,filename +'.jpg')
    return img

#@jit(nopython=True)
def renderData(pos,R,v,resolution,drumDiameter,rpm,fps,displacement,kernels):

    pos_r=np.sqrt(pos[:,0]**2+pos[:,2]**2)
    pos_phi=np.arctan2(pos[:,0],pos[:,2])
    
    omega_drum = rpm/60*2*np.pi
    
    u_drum = omega_drum*pos_r*np.cos(pos_phi)
    v_drum = -omega_drum*pos_r*np.sin(pos_phi)
    
    t=pos[:,1]
    
    th=0.1
    v_front=v[t<th,:]
    pos_front=pos[t<th,:]
    r_front=R[t<th]
    u_drum_front=u_drum[t<th]
    v_drum_front=v_drum[t<th]
    
    u_drum_front_rel=v_front[:,0]-u_drum_front
    v_drum_front_rel=v_front[:,2]-v_drum_front

    pixel=resolution
    u_img=np.zeros((pixel,pixel))
    v_img=np.zeros((pixel,pixel))
    N_img=np.zeros((pixel,pixel))
    drumD=drumDiameter
    
    for ((x,z,y),u_element,v_element,r) in zip(pos_front,u_drum_front_rel,v_drum_front_rel,r_front):
        px=int((-x+drumD/2)/drumD*pixel)
        py=int((-y+drumD/2)/drumD*pixel)
        h=int(r/drumD*pixel)*2+1
        hi=int((h-1)/2)

        kernel=getkernel(kernels,h)
        u_img[py-hi:py+hi+1,px-hi:px+hi+1]+=kernel*u_element
        v_img[py-hi:py+hi+1,px-hi:px+hi+1]+=kernel*v_element
        N_img[py-hi:py+hi+1,px-hi:px+hi+1]+=kernel
    
    N_img = np.where(N_img==0, 1, N_img) 

    u_img=-u_img/N_img
    v_img=-v_img/N_img

    if(displacement):
        scale = 1/fps*resolution/drumDiameter
        u_img=u_img*scale
        v_img=v_img*scale
    return u_img,v_img,N_img

def doPara(file):
    pixel=800
    drumDiameter=0.103
    rpm=8
    fps=FPS
    return readVTKDrumFile(file,pixel,drumDiameter,rpm,fps), file

def f(q,nproc, files): 
    pool = mp.Pool(processes=nproc)
    N=len(files)
    for i in range(math.ceil(N/nproc)):
        while not q.qsize()<3*nproc:
            time.sleep(0.1)
        data = pool.map(doPara,files[i*nproc:i*nproc+nproc])
        for d in data:
            q.put(d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-path', help='an integer for the accumulator',default='DEM/post')
    
    try:
        os.mkdir("analysis")
    except Exception as e:
        pass

    args = parser.parse_args()
    
    data_list = []
    area_list = []
    tracked_objs=[]
    overall_objs=[]
    last_cnt_list=[]
    last_cnt_list_pre=[]
    last_cnt_list_trans=[]
    last_cnt_list_pre_trans=[]
    objcount = 0
    already_analyzed =   0
    vel_shape = None
    
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    log.info("Analysis started at {time}".format( time = current_time))
    
    
    # Check if analysis was aready running
    if os.path.exists('analysis/prev.pickle'):
        #Load previous results to continue
        with open('analysis/prev.pickle',"rb") as file:
            saveObj = pickle.load(file)
            log.info("prev File found and state restored from file")
            (OFFSET_ANALYZE, already_analyzed, tracked_objs,overall_objs,last_cnt_list,last_cnt_list_pre,last_cnt_list_trans,last_cnt_list_pre_trans,objcount,vel_shape,data_list,area_list) = saveObj
            log.info('already_analyzed: '+str(already_analyzed))

    
    
    
    
    
    #read progress info file from liggghts
    with open('info.dat','r') as info_file:
        lines = info_file.readlines()
    
    # Check if all vtk files are available and order them
    for tries in range(5):
        simsteps = [{'timestep':int(x.split()[0]), 'frame':int(x.split()[1])} for x in lines]
        files=glob.glob(args.path+'/rot*liggghts.vtk')
        #log.info(files)
        
        files_simsorted = []
        filetimesteps = [ int(re.findall("[0-9]{10}",x)[-1]) for x in files]
        
        log.info('length simsteps: '+ str(len(simsteps)))
        log.info('length vtk files: '+str(len(files)))
        
        notFoundSteps = []
        for step in simsteps:
            try:
                fileindex = filetimesteps.index(step['timestep'])
                files_simsorted.append(files[fileindex])
                #log.info('timestep: '+str(step['timestep'])+" found at index: "+str(fileindex))
            except Exception:
                notFoundSteps.append(step)
                #log.info('timestep: '+str(step['timestep'])+" not found")
            
        if len(notFoundSteps)==0:
            log.info("All Files found")
            break
        else:
            files_simsorted=[]
            log.info("Not found steps: "+str(notFoundSteps))
            log.info("Retry wait 2s")
            time.sleep(2)
    
            
    #log.info(files_simsorted)
    #remove already analyzed files from list
    filesAnalyze=files_simsorted[already_analyzed:]
    
    log.info('OFFSET_ANALYZE: '+str(OFFSET_ANALYZE))
    log.info('already_analyzed: '+str(already_analyzed))

    # create queue for datahandling in parallel step
    q = mp.Queue()
    # create parallel pool for rendering and run
    nproc=7
    p = mp.Process(target=f, args=(q,nproc, filesAnalyze,))
    p.start()

    #Number of files to analyze
    N=len(filesAnalyze)
    
    #Serial Block analysis step
    for num in range(N):
        log.info('Run process for num: {num}, already_analyzed: {a_a}'.format(num = num,a_a=already_analyzed))

        #get data from queue
        (u,v,dens), file = q.get()
        
        #clear file content (storage space)
        with open(file,"wb") as f:
        	pickle.dump(file, f)
        
        #check if init finished
        if already_analyzed<OFFSET_ANALYZE:
            log.info('index lower than offsest analyze -> go to next')
            already_analyzed += 1
            continue
        	
        log.info("Run analysis for index: {index}".format(index=already_analyzed))
        vel_shape = u.shape

        dens = np.flip(dens,axis=1)
        u = -np.flip(u,axis=1)
        v = np.flip(v,axis=1)

        gray = (dens/np.max(dens)*255).astype(dtype=np.uint8)
        imgRGB=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
        thr = 1.8         
        ret,image1_thr = cv2.threshold(dens.astype(dtype=np.uint8),thr,255,cv2.THRESH_BINARY)
        

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        mask = cv2.morphologyEx(image1_thr, cv2.MORPH_CLOSE, kernel)==255
        
        #mask = image1_thr==255
        
        u=u*mask
        v=v*mask
        mag=np.sqrt(u**2+v**2)

        imgy=gray
        center_circ=(int(imgy.shape[0]/2),int(imgy.shape[1]/2))
        image2_thr_circ = np.zeros(imgy.shape,dtype=np.uint8)
        radius95 = int(imgy.shape[1]/2*0.95)
        cv2.circle(image2_thr_circ,center_circ,radius95,(255),-1)
            
        mask_circ=np.logical_and(image1_thr==255,image2_thr_circ==255)*255
        mask_circ = mask_circ.astype(np.uint8)
        
        contours_circ, hierarchy_circ = cv2.findContours(mask_circ, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        areas = []
        for cnt_circ in contours_circ:
            areas.append(cv2.contourArea(cnt_circ))
            
        area_list.append(np.max(areas))
        log.debug(np.max(areas))
        img_meas=np.zeros(gray.shape,dtype=np.uint8)
        cv2.drawContours(img_meas, [contours_circ[np.argmax(areas)]], -1, 255, 3)
        
        img_circ2=np.zeros(gray.shape,dtype=np.uint8)
        radius_circ = int(imgy.shape[1]/2*0.9)
        cv2.circle(img_circ2,center_circ,radius_circ,(255),-1)
        image_inter=np.logical_and(img_meas==255,img_circ2==255)*255
        image_inter = image_inter.astype(np.uint8)

        data = np.nonzero(image_inter)
        data_list.append(data)

        dudy=np.gradient(u,axis=0)
        dvdx=np.gradient(v,axis=1)

        thr_motion = 3.5
        
        motion_region = np.array( mag>thr_motion,dtype=np.uint8)
        rotation = (dudy-dvdx)
        
        img_result=imgRGB.copy()
        for cnt in last_cnt_list_trans:
             cv2.drawContours(img_result,[cnt],0,(0,0,255),thickness=1)
             M = cv2.moments(cnt)
             cx=int(M['m10']/M['m00'])
             cy=int(M['m01']/M['m00'])
             cv2.circle(img_result,(cx,cy),7,(255,0,255))
    
    
        for j in range(len(tracked_objs)):
            tracked_objs[j][1]=False
        
        last_cnt_list_pre=last_cnt_list
        last_cnt_list=[]
        
        last_cnt_list_pre_trans=last_cnt_list_trans
        last_cnt_list_trans=[]
    
        contours, hir = cv2.findContours(motion_region, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_TC89_L1)
        
        for cnt in contours:
            if cv2.contourArea(cnt)> 500:
            
                img_result = cv2.drawContours(img_result,[cnt],0,(255,0,0),1)
                M = cv2.moments(cnt)
                area = cv2.contourArea(cnt)
                cx=int(M['m10']/M['m00'])
                cy=int(M['m01']/M['m00'])
                if(debug_write_images):
                    cv2.circle(img_result,(cx,cy),10,(0,0,255))
                
                u_mean = np.nanmean(u[cnt[:,0,1],cnt[:,0,0]])
                v_mean = np.nanmean(v[cnt[:,0,1],cnt[:,0,0]])
                direction = np.arctan2(v_mean,u_mean)*180/np.pi
                cnt_rot=rotateCont(cnt, direction)

                img_meas=np.zeros(u.shape,dtype=np.uint8)
                img_meas=cv2.drawContours(img_meas, [cnt_rot], 0, 255, -1)
                
                plist=np.nonzero(img_meas)

                hist, bin_edges = np.histogram(plist[0],bins=100, density=True)
                cum=np.cumsum(hist*np.diff(bin_edges))

                i1=np.nonzero(cum>0.01)[0][0]
                i2=np.nonzero(cum<0.99)[0][-1]
                h98=bin_edges[i2]-bin_edges[i1]
                
                histw, bin_edgesw = np.histogram(plist[1],bins=100, density=True)
                cumw=np.cumsum(histw*np.diff(bin_edgesw))
                i1w=np.nonzero(cumw>0.01)[0][0]
                i2w=np.nonzero(cumw<0.99)[0][-1]
                w98=bin_edgesw[i2w]-bin_edgesw[i1w]

                img_meas=np.zeros(u.shape,dtype=np.uint8)
                img_meas=cv2.drawContours(img_meas, [cnt], 0, 255, -1)
                plist=np.nonzero(img_meas)
                hist, bin_edges = np.histogram(plist[0],bins=100, density=True)
                cum=np.cumsum(hist*np.diff(bin_edges))
                i1=np.nonzero(cum>0.01)[0][0]
                hue98=bin_edges[i1]
                           
                xr,yr,wr,hr = cv2.boundingRect(cnt_rot)
                if(debug_plotSizemeasure):
                    p1=rotatePoint((xr,bin_edges[i1]),-direction,(cx,cy))
                    p2=rotatePoint((xr+wr,bin_edges[i1]),-direction,(cx,cy))
                    p3=rotatePoint((xr+wr,bin_edges[i2]),-direction,(cx,cy))
                    p4=rotatePoint((xr,bin_edges[i2]),-direction,(cx,cy))
                    
                    img_result=cv2.line(img_result,(int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])),(0,100,255),2)
                    img_result=cv2.line(img_result,(int(p3[0]),int(p3[1])),(int(p4[0]),int(p4[1])),(0,100,255),2)
                    
                    p1=rotatePoint((xr,yr),-direction,(cx,cy))
                    p2=rotatePoint((xr+wr,yr),-direction,(cx,cy))
                    p3=rotatePoint((xr+wr,yr+hr),-direction,(cx,cy))
                    p4=rotatePoint((xr,yr+hr),-direction,(cx,cy))
                    
                    img_result=cv2.line(img_result,(int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])),(255,255,0),2)
                    img_result=cv2.line(img_result,(int(p3[0]),int(p3[1])),(int(p4[0]),int(p4[1])),(255,255,0),2)
                
                
                mask = np.zeros((motion_region.shape[0],motion_region.shape[1]),dtype=np.uint8)
                mask = cv2.drawContours(mask,[cnt],0,1,-1)
                
                mask = mask>0
                ublock=u[mask]
                vblock=v[mask]
                umean = np.nanmean(ublock)
                vmean = np.nanmean(vblock)
                uvmagmean=np.sqrt(umean**2+vmean**2)
                rotationmean = np.mean(rotation[mask])
                
                ilist=mask.nonzero()[0]
                jlist=mask.nonzero()[1]
                ublock[np.isnan(ublock)]=0
                vblock[np.isnan(vblock)]=0
                newjList=np.array(ublock+jlist)
                newiList=np.array(vblock+ilist)
                
                pts=np.transpose(np.vstack((jlist,ilist)))
                newpts=np.transpose(np.vstack((newjList,newiList)))
                
                blockrotrate = 0
                
                if len(pts)>1000:
                    trn,trninv = findNonreflectiveSimilarity(pts,newpts)
                    rotphi=np.arctan2(-trn[0,1],trn[0,0])
                    blockrotrate = rotphi*FPS
                    
                
                for (cnt_old,cnt_old_trans) in zip(last_cnt_list_pre,last_cnt_list_pre_trans):
                         M_old = cv2.moments(cnt_old_trans)
                         area_old = cv2.contourArea(cnt_old_trans)
                         cx_old = int(M_old['m10']/M_old['m00'])
                         cy_old = int(M_old['m01']/M_old['m00'])
                         
                         dist = np.sqrt((cx-cx_old)**2+(cy-cy_old)**2)
                         area_ratio = area/area_old
                         match = cv2.matchShapes(cnt,cnt_old_trans,1,0.0)

                         if(dist<100 and area_ratio>0.5 and area_ratio<1.5 and match<1):
                             img_result=cv2.line(img_result,(cx,cy),(cx_old,cy_old),(255,255,0),5)
                             found = False
                             for j in range(len(tracked_objs)):
                                 obj = tracked_objs[j]

                                 if(obj[0][-1].shape==cnt_old.shape and (obj[0][-1]==cnt_old).all()):
                                     #log.debug("already tracked")
                                     obj[0].append(cnt)
                                     obj[2].append(already_analyzed+OFFSET_ANALYZE)
                                     obj[1] = True
                                     obj[3].append((wr,hr,h98,area,w98,hue98))
                                     obj[5].append((uvmagmean,umean,vmean,direction,rotationmean,blockrotrate))
                                     if(obj[4]==-1):
                                         obj[4]=objcount
                                         objcount = objcount+1
                                     if(debug_write_images):
                                        cv2.putText(img_result,str(obj[4]), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
                                        img_result = cv2.drawContours(img_result,[cnt],0,(0,255,255),8)
                         
                                     found = True
                                     
                             if found == False and cy<350:
                                 obj = [[cnt], True,[already_analyzed+OFFSET_ANALYZE],[(wr,hr,h98,area,w98,hue98)],-1,[(uvmagmean,umean,vmean,direction,rotationmean,blockrotrate)]]
                                 tracked_objs.append(obj)
                                 if(debug_write_images):
                                     cv2.putText(img_result,str(obj[4]), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3) 

                
                
                
                if len(pts)>1000:
                    #trn,trninv = findNonreflectiveSimilarity(pts,newpts)
                    posx=np.mean(newpts[:,0])
                    posy=np.mean(newpts[:,1])
                    s=np.sqrt(trn[0,0]**2+trn[1,0]**2)
                    Ttranstonull = np.array( [[1,0,-posx],[0,1,-posy],[0,0,1]])
                    Ttransfromnull = np.array( [[1,0,posx],[0,1,posy],[0,0,1]])
                    Tscale = np.array( [[1/s,0,0],[0,1/s,0],[0,0,1]])
                    
                    coords = cnt[:,0]
                   
                    new_coords = []
                   
                    for c in coords:
                        t=np.hstack((c,1)).dot(trn)
                        t=Ttransfromnull.dot(Tscale).dot(Ttranstonull).dot(t)
                        new_coords.append(t[0:2])
                    
                    new_coords=np.array(new_coords,dtype=np.int32)
                    new_coords=new_coords.reshape((new_coords.shape[0],1,new_coords.shape[1]))
                    if(debug_write_images):
                        cv2.drawContours(img_result,[new_coords],0,(0,255,0),thickness=1)
                    last_cnt_list_trans.append(new_coords)
                    last_cnt_list.append(cnt)
    
        if(debug_write_images):
            cv2.imwrite(datafolder+"/debug/frame"+str(num).zfill(4)+".jpg",img_result)
            log.debug('write debug image')            
            log.debug(datafolder+"/debug/frame"+str(num).zfill(4)+".jpg")

        indexes = []

        for y in range(len(tracked_objs)):        
            if(tracked_objs[y][1]==False):
                L=len(tracked_objs[y][0])
                if L>1:
                    overall_objs.append(tracked_objs[y])
                log.debug(len(tracked_objs))
                indexes.append(y)

        for index in sorted(indexes, reverse=True):
                tracked_objs.pop(index)
        already_analyzed += 1
 
  
    with open('analysis/prev.pickle',"wb") as f:
        saveObj = (OFFSET_ANALYZE, already_analyzed, tracked_objs,overall_objs,last_cnt_list,last_cnt_list_pre,last_cnt_list_trans,last_cnt_list_pre_trans,objcount,vel_shape,data_list,area_list)
        pickle.dump(saveObj, f)      

    ar=1
    ar_list = np.array([1])
    results = {"N":already_analyzed}
    if already_analyzed>=OFFSET_ANALYZE:
        radius95 = int(vel_shape[1]/2*0.95)
        ar=np.mean(area_list)/(radius95*radius95*np.pi)
        ar_list = np.array(area_list)/(radius95*radius95*np.pi)
        
        results["AR_S"] = {"AR":ar}
        results.update(script.InterfaceAnalyzer.analyzeInterface(data_list,area_list))
        
        
    
    with open("analysis/blocks1.pickle","wb") as f:
        infos = {'N':N, 'OFFSET_ANALYZE':OFFSET_ANALYZE, 'area_ratio':ar_list}
        saveObject = (overall_objs,vel_shape, infos)
        pickle.dump(saveObject, f)          
            
        p.join()
    
    if vel_shape != None:
        blockresults = script.CalculateDistributionBlock.calculateDistributions(["analysis/blocks1.pickle"])
        if type(blockresults) == dict:
            results.update(blockresults)

    
    with open("result.pickle","wb") as f:
        pickle.dump(results, f)
        
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    log.info("Analysis finished at {time}".format( time = current_time))



