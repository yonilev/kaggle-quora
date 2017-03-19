from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from data_handler import *
import pickle
import numpy as np


class HashTokenizer(Tokenizer):
    def __init__(self, nb_words,hash_vec_size):
        super(HashTokenizer, self).__init__(nb_words)
        self.hash_vec_size = hash_vec_size

    #overide method
    def texts_to_sequences_generator(self, texts):
        for text in texts:
            seq = text if self.char_level else text_to_word_sequence(text, self.filters, self.lower, self.split)
            vect = [self.get_word_index(word) for word in seq]
            yield vect


    def get_hashed_index(self,word):
        if self.nb_words:
            hash_offset = self.nb_words + 1
        else:
            hash_offset = len(self.word_index)+1

        return (hash(word) % self.hash_vec_size) + hash_offset


    def get_word_index(self,word):
        i = self.word_index.get(word)

        # OOV
        if not i:
            i = self.get_hashed_index(word)

        # MORE THAN FIX SIZE INDEX
        elif self.nb_words and i>self.nb_words:
            i = self.get_hashed_index(word)

        return i


def test_hash_tokenizer():
    tokenizer = HashTokenizer(None,1)
    texts = ['a a b c c d', 'e e f g']
    tokenizer.fit_on_texts(texts)

    for w in 'abcdefghiopkml':
        print w,tokenizer.get_word_index(w)


def save(tokenizer,file_name):
    pickle.dump( tokenizer, open( file_name, 'wb' ) )


def load(file_name):
    return pickle.load( open( file_name, 'rb' ) )


def fit_tokenizer(nb_words,hash_vec_size,file_name):
    tokenizer = HashTokenizer(nb_words, hash_vec_size)
    df = read_train()
    tokenizer.fit_on_texts(df.question1)
    tokenizer.fit_on_texts(df.question2)
    save(tokenizer,file_name)


def main():
    # fit_tokenizer(20000,10000,'tokenizer_20k_10k.p')
    pass

if __name__ == "__main__":
    main()