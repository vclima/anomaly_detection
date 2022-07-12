from struct import iter_unpack
import numpy as np


def binOpen(fileName):
    filePointer=open(fileName,'rb')
    contentBin=filePointer.read(4)
    content=list(iter_unpack('H',contentBin))
    fig_shape=(content[1][0],content[0][0])

    contentBin=filePointer.read(2*fig_shape[0]*fig_shape[1])
    content=list(iter_unpack('H',contentBin))

    image=np.array(content).squeeze()
    
    image=np.reshape(image,fig_shape)

    print(image)
    return image