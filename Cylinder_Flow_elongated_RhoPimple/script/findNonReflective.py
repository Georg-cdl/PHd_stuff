
"""
Created on Wed Jun 10 10:35:09 2020

@author: tobias
"""
import numpy as np

from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank

def findNonreflectiveSimilarity(uv, xy, options=None):
    
    options = {'K': 2}

    K = options['K']
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    # print '--->x, y:\n', x, y

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))
    # print '--->X.shape: ', X.shape
    # print 'X:\n', X

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))
    # print '--->U.shape: ', U.shape
    # print 'U:\n', U

    # We know that X * r = U
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U,rcond=None)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')

    # print '--->r:\n', r

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss,  sc, 0],
        [tx,  ty, 1]
    ])

    # print '--->Tinv:\n', Tinv

    T = inv(Tinv)
    # print '--->T:\n', T

    T[:, 2] = np.array([0, 0, 1])

    return T, Tinv

 

if __name__ == "__main__":
    from_pt = np.array(((1,1),(1,2),(2,2),(2,1))) # a 1x1 rectangle
    to_pt = np.array(((4,4),(6,6),(8,4),(6,2)))   # scaled x 2, rotated 45 degrees and translated
    
    from_pt = np.array(((1,1),(1,2),(2,2),(2,1))) # a 1x1 rectangle
    #to_pt = np.array(((3,1),(3,2),(4,2),(4,1)))   # scaled x 2, rotated 45 degrees and translated
    
    from_pt = np.array(((5,15),(15,15),(15,35),(5,35))) # a 1x1 rectangle
    to_pt = np.array(((15.73,3.21),(20.6,-5.5),(38.11,4.19),(33,12.82)))   # scaled x 2, rotated 45 degrees and translated
    
    from_pt = np.array(((5,15),(15,15),(15,35),(5,35))) # a 1x1 rectangle
    to_pt = np.array(((-5.04,43.82),(1.25,36.03),(16.74,48.57),(10.52,56.33)))   # scaled x 2, rotated 45 degrees and translated
   
    from_pt = np.array(((5,15),(15,15),(15,35),(5,35))) # a 1x1 rectangle
    to_pt = np.array(((0.9,32.41),(5.35,30.15),(9.88,39.03),(5.45,41.34)))   # scaled x 2, rotated 45 degrees and translated
   
    
    trn,trninv = findNonreflectiveSimilarity(from_pt ,to_pt)
    
    print( "Transformation is:")
    print( trn)
    s=np.sqrt(trn[0,0]**2+trn[1,0]**2)
        
    theta = np.arctan2(trn[1,0],trn[0,0])
       
        
    tx = trn[2,0]
    ty = trn[2,1]
    
    posx=np.mean(from_pt[:,0])
    posy=np.mean(from_pt[:,1])

    Tprot=np.linalg.inv(np.array([[1-np.cos(theta) ,-np.sin(theta)],[np.sin(theta), 1-np.cos(theta)]]))     
    t0=Tprot.dot((tx-(1-s)*(posx*np.cos(theta)+posy*np.sin(theta)),ty-(1-s)*(posy*np.cos(theta)-posx*np.sin(theta))))
    
    #x0=30
    #y0=40
    
    #tx1=-x0+x0*np.cos(theta)+y0*np.sin(theta)
    #ty1=-y0+y0*np.cos(theta)-x0*np.sin(theta)
    
    
    
    
    print([s, theta/np.pi*180, tx, ty])
    Tscale = np.array( [[s,0,0],[0,s,0],[0,0,1]])
    Trot = np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])
    Ttrans = np.array( [[1,0,tx],[0,1,ty],[0,0,1]])
    print(Ttrans.dot(Trot).dot(Tscale))
    
    err = 0.0
    for i in range(len(from_pt)):
        fp = from_pt[i]
        tp = to_pt[i]
        result=np.hstack((fp,1)).dot(trn)
        result=Ttrans.dot(Trot).dot(Tscale).dot(np.hstack((fp,1)))
        
        
        print ("%s => %s ~= %s" % (fp, tuple(result[0:2]), tp))
       # err += ((tp[0] - t[0])**2 + (tp[1] - t[1])**2)**0.5
    
    print( "Fitting error = %f" , err)