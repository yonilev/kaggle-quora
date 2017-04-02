from abc import abstractmethod

from keras.layers import Input, LSTM, Dense,Embedding,Bidirectional,Dropout,\
    Conv1D,MaxPool1D,Flatten
from keras.layers.merge import add,concatenate,multiply,maximum
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from hash_tokenizer import *
from embedding import embeddings_for_tokenizer
import math
import json


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
            hidden = Dense(self.params.dense_dim,activation='relu',kernel_regularizer=l2(self.params.l2_dense))(hidden)
            hidden = Dropout(self.params.dropout)(hidden)

        predictions = Dense(1,activation='sigmoid',
                            kernel_regularizer=l2(self.params.l2_dense))(hidden)
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
                            embeddings_regularizer=l2(self.params.l2_embedding),
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
        input1 = pad_sequences(sequences1, maxlen=self.params.seq_length)
        sequences2 = self.tokenizer.texts_to_sequences(df.question2)
        input2 = pad_sequences(sequences2, maxlen=self.params.seq_length)
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

    def predict(self,df,batch_size=1000):
        steps = self.num_of_steps(df,batch_size)
        gen = self.inputs_generator(df,False,batch_size=batch_size)
        preds = self.model.predict_generator(gen,steps,verbose=True)
        return [p[0] for p in preds]


class LSTMSiamese(Siamese):
    def siamese_layers(self, x):
        h = x
        for _ in range(self.params.lstm_layers-1):
           h = Bidirectional(LSTM(self.params.lstm_dim, return_sequences=True, kernel_regularizer=l2(self.params.l2_siamese)))(h)
        return Bidirectional(LSTM(self.params.lstm_dim, kernel_regularizer=l2(self.params.l2_siamese)))(h)

    def generate_params(self,random_params):
        return LSTMParams(random_params)


class CNNSiamese(Siamese):
    def siamese_layers(self, x):
        features = list()
        for kernel_size in range(self.params.min_kernel,self.params.max_kernel+1):
            pool_length = self.params.seq_length - kernel_size + 1
            filters = min(200,self.params.filters*kernel_size)
            conv = Conv1D(filters,kernel_size,activation='tanh',kernel_regularizer=l2(self.params.l2_siamese))(x)
            pool = MaxPool1D(pool_length)(conv)
            feature = Flatten()(pool)
            features.append(feature)
        return concatenate(features)

    def generate_params(self,random_params):
        return CNNParams(random_params)


class Params(object):
    def __init__(self,random_params):
        self.seq_length = 50
        self.dense_layers = 2
        self.dense_dim = 100
        self.embedding_dim = 100
        self.batch_size = 128
        self.l2_embedding = 0.000001
        self.l2_siamese = 0.000001
        self.l2_dense = 0.01
        self.lr = 0.001
        self.dropout = 0.1
        self.pre_train_embedding = 1
        self.tokenizer = None

        if random_params:
            pass


    def __str__(self):
        return str(json.dumps(self.__dict__))


class LSTMParams(Params):
    def __init__(self,random_params):
        super(LSTMParams, self).__init__(random_params)
        self.model = 'lstm'
        self.lstm_layers = 1
        self.lstm_dim = 100
        self.seq_length = 30

        if random_params:
            pass


class CNNParams(Params):
    def __init__(self,random_params):
        super(CNNParams, self).__init__(random_params)
        self.model = 'cnn'
        self.min_kernel = 1
        self.max_kernel = 3
        self.filters = 50
        self.seq_length = 50

        if random_params:
            pass


def load_from_file(prefix,tokenizer_file):
    tokenizer = load(tokenizer_file)
    with open('models/{}.params'.format(prefix)) as f:
        params_dic = json.load(f)

    if params_dic['model']=='cnn':
        params = CNNParams(False)
        params.__dict__ = params_dic
        o = CNNSiamese(tokenizer,params)
    else:
        params = LSTMParams(False)
        params.__dict__ = params_dic
        o = LSTMSiamese(tokenizer,params)
    o.model.load_weights('models/{}.weights'.format(prefix))
    return o


def main():
    pass


if __name__ == "__main__":
    main()

