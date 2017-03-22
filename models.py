from abc import abstractmethod

import keras
from keras.layers import Input, LSTM, Dense,Embedding,Bidirectional
from keras.layers.merge import add,concatenate,multiply,maximum
from keras.models import Model,Sequential
from keras.preprocessing.sequence import pad_sequences
from hash_tokenizer import *


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
            hidden = Dense(self.params.dense_dim,activation='relu')(hidden)

        predictions = Dense(1,activation='sigmoid')(hidden)
        model = Model(inputs=[input1,input2], outputs=predictions)
        return model

    def _get_shared_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.tokenizer.get_input_dim(),
                            output_dim=self.params.embedding_dim,
                            input_length=self.params.seq_length))
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

    def inputs_generator(self,df,batch_size,train=True):
        l = len(df)
        while True:
            for ndx in range(0, l, batch_size):
                curr_batch =  df[ndx:min(ndx + batch_size, l)]
                sequences1 = self.tokenizer.texts_to_sequences(curr_batch.question1)
                input1 = pad_sequences(sequences1, maxlen=self.params.seq_length, truncating='post')
                sequences2 = self.tokenizer.texts_to_sequences(curr_batch.question2)
                input2 = pad_sequences(sequences2, maxlen=self.params.seq_length, truncating='post')

                labels = None
                if train:
                    labels = curr_batch.is_duplicate

                yield [input1,input2],labels




class LSTMSiamese(Siamese):

    def add_siamese_layers(self, model):
        for _ in range(self.params.siamese_layers-1):
            model.add(Bidirectional(LSTM(self.params.lstm_dim, return_sequences=True)))
        model.add(Bidirectional(LSTM(self.params.lstm_dim)))


class Params(object):
    def __init__(self):
        self.seq_length = 50
        self.dense_layers = 1
        self.dense_dim = 50
        self.embedding_dim = 50
        self.siamese_layers = 1
        self.lstm_dim = 50
        self.batch_size = 64



def main():
    pass


if __name__ == "__main__":
    main()

