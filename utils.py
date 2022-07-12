from struct import iter_unpack,unpack
import numpy as np
import cv2


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

def normalize(img):
    out=cv2.normalize(img,None,1,0,cv2.NORM_MINMAX,cv2.CV_32F)
    return out