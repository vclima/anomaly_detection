from struct import iter_unpack,unpack
import numpy as np
import cv2
import denoiser


def binOpen(fileName):
    filePointer=open(fileName,'rb')
    
    contentBin=filePointer.read(2)
    content=unpack('H',contentBin)
    img_type=content[0]
    
    contentBin=filePointer.read(4)
    content=list(iter_unpack('H',contentBin))
    fig_shape=(content[1][0],content[0][0])

    if img_type:
        contentBin=filePointer.read(2*fig_shape[0]*fig_shape[1])
        content=list(iter_unpack('H',contentBin))
        image=np.array(content).squeeze()
        image=np.reshape(image,fig_shape)
        image=np.uint16(image)
        image = normalize(image)
    else: 
        contentBin=filePointer.read(2*fig_shape[0]*fig_shape[1])
        content=list(iter_unpack('H',contentBin))
        image=np.array(content).squeeze()
        image=np.reshape(image,fig_shape)
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
        U, sigmas, V = np.linalg.svd(Y, full_matrices=False);
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