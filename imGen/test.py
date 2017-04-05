import numpy as np
import os
import skimage.io as io
os.environ["KERAS_BACKEND"] = "theano"
from keras.layers import Convolution2D, Dense, Dropout, LeakyReLU,Input,Flatten, Reshape,UpSampling2D
from keras.layers import ZeroPadding2D
from keras.models import Model
from keras.optimizers import Adam

def converToOneHot(lst, num_classes):
    #conver list of ints to list of one hot vectors
    vec = [ [0]*(l-1) + [l] +[0]*(num_classes-l) for l in lst]

    return vec
opt = Adam(lr=1e-3)

def load_dataset(path):
    trainX = []
    trainY = []
    testX = []
    testY = []

    nc = 0
    for name in os.listdir(path):
        #print(os.path.join(path, name))
        if  name[0] != 0:
            print('big img')
            continue
        y = name[1]
        if y > nc: nc = y
        I = io.imread("%s/%s",(path,name))
        if(len(name) >= 8):
            testX.append(I)
            testY.append(y)
        else:
            trainX.append(I)
            trainY.append(y)

        trainY = converToOneHot(trainY,nc)
        testY = converToOneHot(testY,nc)
        print("Done!")
    return (trainX,trainY),(testX,testY)


def buldStyleConv(n_w,n_h):
    inp = Input(shape=(3,n_w,n_h))
    d = Convolution2D(256*3,5,5,dim_ordering='th',border_mode='same', activation='relu')(inp)
    #d = LeakyReLU(0.2)(d)
    #d = Dropout(0.23)(d)
    d = Convolution2D(256*3*2,5,5,dim_ordering='th',border_mode='same', activation='relu')(d)
    #d = LeakyReLU(0.2)(d)
    #d = Dropout(0.23)(d)
    d = Flatten()(d)
    d = Dense(256*3,activation='relu')(d)
    d = LeakyReLU(0.2)(d)
    d = Dropout(0.25)(d)
    d_out = Dense(8,activation='softmax')(d)

    return Model(inp,d_out)

discr = buldStyleConv(256,180)
discr.compile(opt,'categorical_crossentropy')
discr.summary()


