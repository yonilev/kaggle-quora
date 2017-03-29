from abc import abstractmethod

from keras.layers import Input, LSTM, Dense,Embedding,Bidirectional,Dropout,\
    Activation,BatchNormalization,Conv1D,MaxPool1D,Flatten
from keras.layers.merge import add,concatenate,multiply,maximum
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from hash_tokenizer import *
from embedding import embeddings_for_tokenizer
import math
import json
import random


class Siamese(object):
    def __init__(self,tokenizer,random_params=False,verbose=False):
        self.tokenizer = tokenizer
        self.params = self.generate_params(random_params)
        self.params.tokenizer = self.tokenizer.path
        self.verbose = verbose
        self.model = self._create_model()

        if self.verbose:
            print (self.model.summary())

    def _create_model(self):
        input_shape = (self.params.seq_length,)
        input1 = Input(shape=input_shape)
        input2 = Input(shape=input_shape)

        shared_model = self._get_shared_model(input_shape)
        if self.verbose:
            print (shared_model.summary())

        hidden1 = shared_model(input1)
        hidden2 = shared_model(input2)

        hidden = self._merge_sides(hidden1,hidden2)

        for _ in range(self.params.dense_layers):
            hidden = Dense(self.params.dense_dim,kernel_regularizer=l2(self.params.l2))(hidden)
            if self.params.batch_norm:
                hidden = BatchNormalization()(hidden)
            hidden = Activation('relu')(hidden)
            hidden = Dropout(self.params.dropout)(hidden)

        predictions = Dense(1,activation='sigmoid',
                            kernel_regularizer=l2(self.params.l2))(hidden)
        model = Model(inputs=[input1,input2], outputs=predictions)
        return model

    def _get_shared_model(self,input_shape):
        input1 = Input(input_shape)

        if self.params.pre_train_embedding:
            weights = embeddings_for_tokenizer(self.tokenizer,self.params.embedding_dim)
            weights = [weights]
        else:
            weights = None

        embedding = Embedding(input_dim=self.tokenizer.get_input_dim(),
                            output_dim=self.params.embedding_dim,
                            input_length=self.params.seq_length,
                            embeddings_regularizer=l2(self.params.l2),
                            weights=weights)(input1)
        outputs = self.siamese_layers(embedding)
        return Model(inputs=input1,outputs=outputs)


    @staticmethod
    def _merge_sides(hidden1, hidden2):
        merges = list()
        merges.append(add([hidden1,hidden2]))
        merges.append(multiply([hidden1,hidden2]))
        merges.append(maximum([hidden1,hidden2]))
        return concatenate(merges)

    @abstractmethod
    def siamese_layers(self, x):
        """
        add siamese layers to model, e.g., lstm.
        """

    @abstractmethod
    def generate_params(self,random_params):
        """
        generate model params
        """

    def create_inputs(self,df):
        sequences1 = self.tokenizer.texts_to_sequences(df.question1)
        input1 = pad_sequences(sequences1, maxlen=self.params.seq_length, truncating='post')
        sequences2 = self.tokenizer.texts_to_sequences(df.question2)
        input2 = pad_sequences(sequences2, maxlen=self.params.seq_length, truncating='post')
        return [input1,input2]

    def inputs_generator(self,df,train=True,batch_size=None):
        l = len(df)
        if batch_size is None:
            batch_size = self.params.batch_size

        while True:
            for ndx in range(0, l, batch_size):
                curr_batch =  df[ndx:min(ndx + batch_size, l)]
                labels = None
                if train:
                    labels = curr_batch.is_duplicate

                yield self.create_inputs(curr_batch),labels

    def num_of_steps(self,df,batch_size):
        return int(math.ceil(len(df)*1.0/batch_size))

    def predict(self,df,batch_size=10000):
        steps = self.num_of_steps(df,batch_size)
        gen = self.inputs_generator(df,False,batch_size=batch_size)
        preds = self.model.predict_generator(gen,steps)
        return [p[0] for p in preds]


class LSTMSiamese(Siamese):
    def siamese_layers(self, x):
        h = x
        for _ in range(self.params.lstm_layers-1):
           h = Bidirectional(LSTM(self.params.lstm_dim, return_sequences=True, kernel_regularizer=l2(self.params.l2)))(h)
        return Bidirectional(LSTM(self.params.lstm_dim, kernel_regularizer=l2(self.params.l2)))(h)

    def generate_params(self,random_params):
        return LSTMParams(random_params)


class CNNSiamese(Siamese):
    def siamese_layers(self, x):
        features = list()
        for kernel_size in range(self.params.min_kernel,self.params.max_kernel+1):
            pool_length = self.params.seq_length - kernel_size + 1
            filters = min(200,self.params.filters*kernel_size)
            conv = Conv1D(filters,kernel_size,activation='tanh',kernel_regularizer=l2(self.params.l2))(x)
            pool = MaxPool1D(pool_length)(conv)
            feature = Flatten()(pool)
            features.append(feature)
        return concatenate(features)

    def generate_params(self,random_params):
        return CNNParams(random_params)


class Params(object):
    def __init__(self,random_params):
        self.seq_length = 50
        self.dense_layers = 1
        self.dense_dim = 50
        self.embedding_dim = 50
        self.batch_size = 64
        self.l2 = 0.00002
        self.lr = 0.001
        self.batch_norm = 0
        self.dropout = 0.1
        self.pre_train_embedding = 1
        self.tokenizer = None

        if random_params:
            self.seq_length = random.choice([30,50])
            self.dense_layers = random.choice([1,2,3])
            self.dense_dim = random.choice([50,100,300])
            self.embedding_dim = random.choice([50,100,300])
            self.batch_size = random.choice([64,128,256])
            self.l2 = 10**random.uniform(-7,-1)
            self.lr = 10**random.uniform(-5,-2)
            self.batch_norm = random.choice([0,1])
            self.dropout = random.choice([0,0.1,0.5])
            self.pre_train_embedding = random.choice([0,1])


    def __str__(self):
        return str(json.dumps(self.__dict__))


class LSTMParams(Params):
    def __init__(self,random_params):
        super(LSTMParams, self).__init__(random_params)
        self.model = 'lstm'
        self.lstm_layers = 1
        self.lstm_dim = 50

        if random_params:
            self.lstm_layers = random.choice([1,2])
            self.lstm_dim = random.choice([50,100,300])


class CNNParams(Params):
    def __init__(self,random_params):
        super(CNNParams, self).__init__(random_params)
        self.model = 'cnn'
        self.min_kernel = 1
        self.max_kernel = 3
        self.filters = 25

        if random_params:
            self.min_kernel = random.choice([1,2,3])
            self.max_kernel = random.choice([3,4,5])
            self.filters = random.choice([25,50])


def load_from_file(prefix,tokenizer_file):
    tokenizer = load(tokenizer_file)
    with open('models/{}.params'.format(prefix)) as f:
        params_dic = json.load(f)

    if params_dic['model']=='cnn':
        params = CNNParams()
        params.__dict__ = params_dic
        o = CNNSiamese(tokenizer,params)
    else:
        params = LSTMParams()
        params.__dict__ = params_dic
        o = LSTMSiamese(tokenizer,params)
    o.model.load_weights('models/{}.weights'.format(prefix))
    return o


MODEL_PREFIX = 'cnn_baseline'


def main():
    pass

if __name__ == "__main__":
    main()

