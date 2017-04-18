from abc import abstractmethod

from keras.layers import Input, GRU, LSTM, Dense, Embedding, Bidirectional, Dropout, \
    Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from hash_tokenizer import *
from embedding import embeddings_for_tokenizer
from attention import *
from features_extraction import FeaturesExtractor
import math
import json


class RNNSiamese(object):
    def __init__(self, tokenizer, random_params=False, verbose=False):
        self.tokenizer = tokenizer
        self.params = Params(random_params)
        self.params.tokenizer = self.tokenizer.path
        self.verbose = verbose
        self.features_extractor = FeaturesExtractor(tokenizer)
        self.model = self._create_model()

        if self.verbose:
            self.model.summary()

    def _create_model(self):
        input_shape = (self.params.seq_length,)
        input1 = Input(shape=input_shape)
        input2 = Input(shape=input_shape)

        shared_model = self._get_shared_model(input_shape)
        if self.verbose:
            shared_model.summary()

        hidden1 = shared_model(input1)
        hidden2 = shared_model(input2)
        hidden = Lambda(abs_diff, output_shape=(shared_model.output_shape[1],))([hidden1, hidden2])

        input3 = Input(shape=(self.features_extractor.number_of_features(),))
        hidden = concatenate([hidden,input3])

        for _ in range(self.params.dense_layers):
            hidden = Dense(self.params.dense_dim, activation='relu', kernel_regularizer=l2(self.params.l2_dense))(
                hidden)
            hidden = Dropout(self.params.dropout)(hidden)

        predictions = Dense(1, activation='sigmoid',
                            kernel_regularizer=l2(self.params.l2_dense))(hidden)
        model = Model(inputs=[input1, input2,input3], outputs=predictions)
        return model

    def _get_shared_model(self, input_shape):
        input1 = Input(input_shape)

        if self.params.pre_train_embedding:
            weights = embeddings_for_tokenizer(self.tokenizer, self.params.embedding_dim)
            weights = [weights]
        else:
            weights = None

        embedding = Embedding(input_dim=self.tokenizer.get_input_dim(),
                              output_dim=self.params.embedding_dim,
                              input_length=self.params.seq_length,
                              embeddings_regularizer=l2(self.params.l2_embedding),
                              weights=weights, mask_zero=True)(input1)
        outputs = self.siamese_layers(embedding,self.params.attention)
        return Model(inputs=input1, outputs=outputs)

    def siamese_layers(self, x, attention):
        h = Bidirectional(LSTM(self.params.rnn_dim,
                               return_sequences=attention,
                               kernel_regularizer=l2(self.params.l2_siamese),
                               recurrent_regularizer=l2(self.params.l2_siamese)))(x)

        if attention:
            return AttentionWithContext(W_regularizer=l2(self.params.l2_siamese),
                                        u_regularizer=l2(self.params.l2_siamese))(h)
        return h

    def create_inputs(self, df):
        sequences1 = self.tokenizer.texts_to_sequences(df.question1)
        input1 = pad_sequences(sequences1, maxlen=self.params.seq_length)
        sequences2 = self.tokenizer.texts_to_sequences(df.question2)
        input2 = pad_sequences(sequences2, maxlen=self.params.seq_length)
        return [input1, input2]

    def inputs_generator(self, df, train=True, batch_size=None):
        l = len(df)
        if batch_size is None:
            batch_size = self.params.batch_size

        df['features'] = self.features_extractor.extract(df)

        while True:
            for ndx in range(0, l, batch_size):
                curr_batch = df[ndx:min(ndx + batch_size, l)]
                labels = None
                if train:
                    labels = curr_batch.is_duplicate

                input1, input2 = self.create_inputs(curr_batch)
                input3 = np.array(curr_batch.features.tolist())
                yield [input1,input2,input3], labels

    @staticmethod
    def num_of_steps(df, batch_size):
        return int(math.ceil(len(df) * 1.0 / batch_size))

    def predict(self, df, batch_size=1000):
        steps = self.num_of_steps(df, batch_size)
        gen = self.inputs_generator(df, False, batch_size=batch_size)
        preds = self.model.predict_generator(gen, steps, verbose=True)
        return [p[0] for p in preds]


def abs_diff(hiddens):
    hidden1, hidden2 = hiddens
    return K.abs(hidden1 - hidden2)


class Params(object):
    def __init__(self, random_params):
        self.seq_length = 30
        self.dense_layers = 1
        self.dense_dim = 100
        self.embedding_dim = 300
        self.batch_size = 64
        self.l2_embedding = 1e-7
        self.l2_siamese = 1e-4
        self.l2_dense = 1e-4
        self.lr = 0.001
        self.dropout = 0.1
        self.pre_train_embedding = 1
        self.tokenizer = None
        self.clipnorm = 1
        self.rnn_dim = 50
        self.attention = False

        if random_params:
            pass

    def __str__(self):
        return str(json.dumps(self.__dict__))


def load_from_file(prefix, tokenizer_file):
    tokenizer = load(tokenizer_file)
    with open('models/{}.params'.format(prefix)) as f:
        params_dic = json.load(f)

    params = Params(False)
    params.__dict__ = params_dic
    o = RNNSiamese(tokenizer, params)
    o.model.load_weights('models/{}.weights'.format(prefix))
    return o


def main():
    pass


if __name__ == "__main__":
    main()
