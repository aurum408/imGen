import os
import json
#print(keras.backend.backend())
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
import numpy as np


#passes is dictionary with passes
#{'type_of data'-pass}
class data_loader(object):
    def _init_(self, passes):
        self.pathToCaps = passes['caps']
        self.pathToImgSmall = passes['small_img']
        self.pathToImgBig = passes['big_img']
        self.pathToEmbedding = passes['embedding']
        self.MAX_NB_WORDS = 20000
        self.MAX_CAP_LEN = 20
        self.EMBEDDING_DIM = 300

    def load_text(self,path):
        print ("Start!")
        caps = []  # captions
        ids = []  # images
        keys = {}  # image - all it's captions (ids)
        c = 0
        for name in os.listdir(path):
            # print(os.path.join(path, name))
            if os.path.isdir(os.path.join(path, name)):
                print('dir')
                continue
            lst = []
            ids.append(name[:-5])

            with open('%s/%s' % (path, name)) as fp:
                data = json.load(fp)

            for n in range(len(data)):
                caps.append(data[n])

            keys[name[:-5]] = [i for i in range(len(caps) - len(data), len(caps))]
            c = c + 1

        print("Done!")
        return [caps, ids, keys]

    def preprocess(self):
        #load all caps in a dict
        ds = {}
        n=0
        for path in self.pathToCaps:
            ds[n] = self.load_text(path)
            n = n+1
        print ('found ', n, 'data folders')
        if n > 1: print ('data from 1 will be ignored')
        vals = ds.values()
        CONST = len(vals[0][0])
        all_caps = vals[0][0] + vals[1][0]
        all_caps = [l.encode('ascii', 'ignore') for l in all_caps]
        vals[1][-1].update((k, map(lambda i: i+CONST, vals[1][-1][k])) for k in vals[1][-1]) # add len of train data to all ids in val)

        tokenizer = Tokenizer(self.MAX_NB_WORDS)
        tokenizer.fit_on_texts(all_caps)
        sequences = tokenizer.texts_to_sequences(all_caps)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        data = pad_sequences(sequences, maxlen=self.MAX_CAP_LEN)
        print('Shape of data tensor:', data.shape)

        embeddings_index = {}
        f = open(self.pathToEmbedding)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))

        embedding_matrix = np.zeros((len(word_index) + 1, self.EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        return embedding_matrix, ds, data, word_index















