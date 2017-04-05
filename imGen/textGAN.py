from keras.layers import Embedding, Dense, Conv2D, UpSampling2D, MaxPooling2D, Input, Flatten
from keras.models import Model
import data_loader


class textGAN(object):
     def __init__(self, img_shape, embd_size, noize_shape, batch_size):
        self.img_shape = img_shape
        self.embd_size = embd_size
        self.batch_size = batch_size
        self.noize_sh = noize_shape



     def buld_gen(self,noise, text):
         inp1 = Input(shape=self.noize_sh,name='noize_inp')
         inp2 = Input(shape=self.embd_size, name='emedding_inp')
         d = Embedding()(inp2)
         d = Flatten()(d)
         d = Dense(600)(d)


