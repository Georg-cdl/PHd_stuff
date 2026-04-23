#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 09:39:31 2020

@author: tobias
https://stackoverflow.com/questions/53406534/procedural-circle-mesh-with-uniform-faces
conda install -c conda-forge numpy-stl 
"""
from scipy.spatial import Delaunay



import numpy as np
from stl import mesh, Mode

# Define the 8 vertices of the cube
vertices_shell = []
vertices_shell2 = []
faces_shell = []
N=6*10
dges=63
n=4
R=103.0/2.0

d=dges/n
for i in range(N):
    
    for j in range(n+1):
        print(j)
        vertices_shell.append([R*np.cos(i/N*np.pi*2), d*j, R*np.sin(i/N*np.pi*2)]) 
   
    
    vertices_shell2.append([R*np.cos(i/N*np.pi*2), R*np.sin(i/N*np.pi*2)]) # 2*i
    
    
    if i>0:
        
        for j in range(n):
            faces_shell.append([(n+1)*i+j, (n+1)*i+1+j, (n+1)*(i-1)+j])
            faces_shell.append([(n+1)*i+1+j, (n+1)*(i-1)+1+j, (n+1)*(i-1)+j])
    else:
        for j in range(n):
            faces_shell.append([(n+1)*i+j, (n+1)*i+1+j, (n+1)*(N-1)+j])
            faces_shell.append([(n+1)*i+1+j, (n+1)*(N-1)+1+j, (n+1)*(N-1)+j])
        #faces_shell.append([2*i, 2*i+1, 2*((N-0)-1)])
        #faces_shell.append([2*i+1, 2*((N-0)-1)+1, 2*((N-0)-1)])
        print(i)
        


#points=np.array(vertices_shell)[0::2,(0,2)]
points=np.array(vertices_shell2)
tri = Delaunay(points)

import matplotlib.pyplot as plt

import triangle as tr


A = dict(vertices=points)
B = tr.triangulate(A, 'cDq20a50')
tr.compare(plt, A, B)

plt.show()



            

vertices_shell = np.array(vertices_shell)
faces_shell = np.array(faces_shell)
  
vertices_cov1 = np.array(B['vertices'])
vertices_cov1 = np.array([np.array(vertices_cov1)[:,0],np.array(vertices_cov1)[:,0]*0,np.array(vertices_cov1)[:,1]]).transpose()

faces_cov1 = np.array(B['triangles'])  
 
vertices_cov2 = np.array(vertices_cov1)+[0, dges, 0]
faces_cov2 = np.array(B['triangles'])  

# # Create the mesh shell
shell = mesh.Mesh(np.zeros(faces_shell.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces_shell):
    for j in range(3):
        shell.vectors[i][j] = vertices_shell[f[j],:]

shell.save('shell.stl', mode=Mode.ASCII)

# # Create the mesh cover1
cov1 = mesh.Mesh(np.zeros(faces_cov1.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces_cov1):
    for j in range(3):
        cov1.vectors[i][j] = vertices_cov1[f[j],:]

cov1.save('cov1.stl', mode=Mode.ASCII)

# # Create the mesh cover1
cov2 = mesh.Mesh(np.zeros(faces_cov2.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces_cov2):
    for j in range(3):
        cov2.vectors[i][j] = vertices_cov2[f[j],:]

cov2.save('cov2.stl', mode=Mode.ASCII)

points=np.array(vertices_cov1)[0::2,(0,2)]
#points=np.array(vertices_shell2)
tri = Delaunay(points)



#%%

