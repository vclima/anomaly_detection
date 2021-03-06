import numpy as np
import ffdnet # pip install ffdnet


# proxl1
def proxl1(x,lamb):
    return np.sign(x)*np.maximum((np.abs(x)-lamb),np.zeros(x.shape))

# proxl2 lamb/2 || . ||_2^2
def proxl2sq(x,lamb):
    return x/(1+lamb)

# ffdnet single image (x.shape=m1,m2)
def ffd(x,sigma):
    # normalize input
    ma=np.max(x)
    mi=np.min(x)
    x=(x-mi)/(ma-mi)
    
    #img_input=np.dstack((img,img,img))
    img_input=x.reshape(x.shape[0],x.shape[1],1)
    denoised=ffdnet.run(img_input,sigma)#[:,:,0]
    denoised=denoised.reshape(x.shape[0],x.shape[1])
    
    # denormalize output
    denoised=denoised*(ma-mi)+mi
    
    return denoised
