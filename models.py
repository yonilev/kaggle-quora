from abc import abstractmethod

from keras.layers import Input, LSTM, Dense,Embedding,Bidirectional,Dropout,Activation,BatchNormalization
from keras.layers.merge import add,concatenate,multiply,maximum
from keras.models import Model,Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from hash_tokenizer import *
import math
import json


class Siamese(object):
    def __init__(self,tokenizer,params):
        self.tokenizer = tokenizer
        self.params = params
        self.model = self._create_model()

    def _create_model(self):
        shared_model = self._get_shared_model()

        input1 = Input(shape=(self.params.seq_length,))
        input2 = Input(shape=(self.params.seq_length,))

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

    def _get_shared_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.tokenizer.get_input_dim(),
                            output_dim=self.params.embedding_dim,
                            input_length=self.params.seq_length,
                            embeddings_regularizer=l2(self.params.l2)))
        self.add_siamese_layers(model)
        return model

    @staticmethod
    def _merge_sides(hidden1, hidden2):
        merges = list()
        merges.append(add([hidden1,hidden2]))
        merges.append(multiply([hidden1,hidden2]))
        merges.append(maximum([hidden1,hidden2]))
        return concatenate(merges)

    @abstractmethod
    def add_siamese_layers(self, model):
        """
        add siamese layers to model, e.g., lstm.
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
    def add_siamese_layers(self, model):
        for _ in range(self.params.siamese_layers-1):
            model.add(Bidirectional(LSTM(self.params.lstm_dim, return_sequences=True)))
        model.add(Bidirectional(LSTM(self.params.lstm_dim)))


class Params(object):
    def __init__(self,random=False):
        self.seq_length = 50
        self.dense_layers = 1
        self.dense_dim = 50
        self.embedding_dim = 50
        self.siamese_layers = 1
        self.lstm_dim = 50
        self.batch_size = 64
        self.l2 = 0.001
        self.lr = 0.001
        self.batch_norm = False
        self.dropout = 0

        if random:
            self.dense_layers = np.random.choice([1,2,3])
            self.dense_dim = np.random.choice([50,100,300])
            self.embedding_dim = np.random.choice([50,100,300])
            self.siamese_layers = np.random.choice([1,2])
            self.lstm_dim = np.random.choice([50,100,300])
            self.batch_size = np.random.choice([64,128,256])
            self.l2 = 10**np.random.uniform(-7,0)
            self.lr = 10**np.random.uniform(-4,-2)
            self.batch_norm = np.random.choice([False,True])
            self.dropout = np.random.choice([0,0.1,0.5])

    def __str__(self):
        return str(json.dumps(self.__dict__))


def load_from_file(prefix,tokenizer_file,model_class):
    tokenizer = load(tokenizer_file)
    params = Params()
    with open('models/{}.params'.format(prefix)) as f:
        params.__dict__ = json.load(f)

    o = model_class(tokenizer,params)
    o.model.load_weights('models/{}.weights'.format(prefix))
    return o


MODEL_PREFIX = 'lstm_baseline2'

def main():
    pass


if __name__ == "__main__":
    main()

