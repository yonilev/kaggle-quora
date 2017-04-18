from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from data_handler import *
from utils import *
import numpy as np


class HashTokenizer(Tokenizer):
    def __init__(self, num_words,hash_vec_size, path):
        super(HashTokenizer, self).__init__(num_words)
        self.hash_vec_size = hash_vec_size
        self.path = path

    #overide method
    def texts_to_sequences_generator(self, texts):
        for text in texts:
            seq = text if self.char_level else text_to_word_sequence(text, self.filters, self.lower, self.split)
            vect = [self.get_word_index(word) for word in seq]
            yield vect


    def get_hashed_index(self,word):
        if self.num_words:
            hash_offset = self.num_words + 1
        else:
            hash_offset = len(self.word_index)+1

        return (hash(word) % self.hash_vec_size) + hash_offset


    def get_word_index(self,word):
        i = self.word_index.get(word)

        # OOV
        if not i:
            i = self.get_hashed_index(word)

        # MORE THAN FIX SIZE INDEX
        elif self.num_words and i>self.num_words:
            i = self.get_hashed_index(word)

        return i

    def get_input_dim(self):
        if self.num_words:
            num_words = self.num_words
        else:
            num_words = len(self.word_index)

        return num_words + self.hash_vec_size + 1

    def is_unknown(self,word):
        return self.get_word_index(word)>self.num_words


def test_hash_tokenizer():
    tokenizer = HashTokenizer(3,1,'')
    texts = ['a a b c c d', 'e e f g']
    tokenizer.fit_on_texts(texts)




def fit_tokenizer(nb_words,hash_vec_size,file_name,df):
    tokenizer = HashTokenizer(nb_words, hash_vec_size,file_name)
    tokenizer.fit_on_texts(np.concatenate([df.question1,df.question2],axis=0))
    save(tokenizer,file_name)


TOKENIZER_20K_1K = 'tokenizers/tokenizer_20k_1k.p'
TOKENIZER_20K_ONE = 'tokenizers/tokenizer_20k_1.p'



def main():
    df_train,_,_ = read_train()

    fit_tokenizer(20000,1000,TOKENIZER_20K_1K,df_train)
    fit_tokenizer(20000,1,TOKENIZER_20K_ONE,df_train)

    # fit_tokenizer(None,1,TOKENIZER_ALL,df_train)

    pass

if __name__ == "__main__":
    main()