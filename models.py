from abc import abstractmethod

import keras
from keras.layers import Input, LSTM, Dense,Embedding,Bidirectional
from keras.layers.merge import add,concatenate,multiply,maximum
from keras.models import Model,Sequential
from hash_tokenizer import *


class Params(object):
    def __init__(self):
        self.seq_length = 50
        self.dense_layers = 2
        self.dense_dim = 300
        self.embedding_dim = 300
        self.siamese_layers = 2
        self.lstm_dim = 300


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


class LSTMSiamese(Siamese):

    def add_siamese_layers(self, model):
        for _ in range(self.params.siamese_layers-1):
            model.add(Bidirectional(LSTM(self.params.lstm_dim, return_sequences=True)))
        model.add(Bidirectional(LSTM(self.params.lstm_dim)))


def main():
    tokenizer = load('tokenizer_20k_10k.p')
    lstm_model = LSTMSiamese(tokenizer,Params())
    lstm_model.model.compile(optimizer='adam',loss='binary_crossentropy')
    print lstm_model.model.summary()

if __name__ == "__main__":
    main()

