from struct import iter_unpack,unpack
import numpy as np
from scipy.sparse.linalg import svds
import cv2
import denoiser
import os
from time import time

from PIL import Image


def binOpen(fileName,debug=False):
    
    filePointer=open(fileName,'rb')
    contentBin=filePointer.read()
    content=list(iter_unpack('H',contentBin))
    img_type=content[0][0]
    if debug:
        print(img_type)
    fig_shape=(content[2][0],content[1][0])
    
    if debug:
        print(fig_shape)

    if img_type:
        c2=[[row[c] for row in content] for c in range(len(content[0]))]
        content=np.array(c2).T
        content=content[:,0]
        content=content[3:]
        image=np.reshape(content,fig_shape)
        image=np.uint16(image)
        image = normalize(image)

    else: 
        c2=[[row[c] for row in content] for c in range(len(content[0]))]
        content=np.array(c2).T
        content=content[:,0]
        content=content[3:]
        image=np.reshape(content,fig_shape)
        image=np.uint16(image)
        image = cv2.cvtColor(image, cv2.COLOR_BayerGR2GRAY)
        image = normalize(image)

    filePointer.close()
    return image,fig_shape

def normalize(img,mi=0,ma=1):
    out=cv2.normalize(img,None,ma,mi,cv2.NORM_MINMAX,cv2.CV_32F)
    return out

def pcp(M, lam=None, mu=None, factor=1, tol=1e-3,maxit=1000,debug=True):
    # initialization
    m, n = M.shape
    unobserved = np.isnan(M)
    M[unobserved] = 0
    S = np.zeros((m,n))
    L = np.zeros((m,n))
    Lambda = np.zeros((m,n)) # the dual variable

    # parameter setting
    if mu is None:
        mu = 0.25/np.abs(M).mean()
    if lam is None:
        lam = 1/np.sqrt(max(m,n)) * float(factor)
        
    print('mu=',mu)
    print('lambda=',lam)
        
    # main
    for k in range(maxit):
        normLS = np.linalg.norm(np.concatenate((S,L), axis=1), 'fro')              
        # dS, dL record the change of S and L, only used for stopping criterion

        X = Lambda / mu + M
        # L - subproblem
        Y = X - S;
        dL = L;  

        # EXPENSIVE SVD
        U, sigmas, V = np.linalg.svd(Y, full_matrices=False);

        # LESS EXPENSIVE SVD
        #pre_rank=max(min(np.linalg.matrix_rank(Y),min(Y.shape)-1),1)
        #U,sigmas,V=svds(Y,k=pre_rank)
        

        rank = (sigmas > 1/mu).sum()
        Sigma = np.diag(sigmas[0:rank] - 1/mu)
        L = np.dot(np.dot(U[:,0:rank], Sigma), V[0:rank,:])
        dL = L - dL
        
        # S - subproblem 
        Y = X - L
        dS = S
        S = denoiser.proxl1(Y, lam/mu) # softshinkage operator 
        dS = S - dS

        # Update Lambda (dual variable)
        Z = M - S - L
        Z[unobserved] = 0
        Lambda = Lambda + mu * Z;
        
        # stopping criterion
        RelChg = np.linalg.norm(np.concatenate((dS, dL), axis=1), 'fro') / (normLS + 1)
        if RelChg < tol: 
            break
            
        # debug
        if debug is True:
            print(k,':',RelChg)
    return L, S, k, rank


def convertBin(infolder,outfolder):

    for k,filename in enumerate(os.listdir(infolder)):
        img=binOpen(filename)
        
        vis= np.around(normalize(img,0,255))

        im = Image.fromarray(vis)
        im=im.convert("L")
        im.save(outfolder+filename+'.jpeg')
       
    return