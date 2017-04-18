import pandas as pd
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from keras.preprocessing.text import text_to_word_sequence


SEED = 1234
TRAIN_FILE = 'data/train.csv'
TEST_FILE = 'data/test.csv'
PREPROCESSED = '_preprocessed'


def read_file(file_name,nrows=None,preprocessed=True):
    if preprocessed:
        file_name = preprocessed_name(file_name)
    df =  pd.read_csv(file_name,nrows=nrows)
    df['question1'] = df.question1.apply(str)
    df['question2'] = df.question2.apply(str)

    return df


def preprocessed_name(file_name):
    return file_name.replace('.csv', '%s.csv' % PREPROCESSED)


def preprocess_files():
    preprocess_file(TRAIN_FILE)
    preprocess_file(TEST_FILE)


def preprocess_file(file_name):
    df = read_file(file_name,preprocessed=False)
    df['question1'] = df.question1.apply(tokenize_text)
    df['question2'] = df.question2.apply(tokenize_text)
    df.to_csv(preprocessed_name(file_name), index=False)


def tokenize_text(text):
    text = text.replace('’',"'")
    sent_tokenizer = PunktSentenceTokenizer()
    sents = sent_tokenizer.tokenize(text)
    l = list()
    for s in sents:
        l.append('mybeginsent')
        l.append(s)
        l.append('myendsent')
    text = ' '.join(l)
    text = ' '.join(text_to_word_sequence(text))
    return ' '.join(word_tokenize(text))


def read_train(test_size=0.1,val_size=0.5,nrows=None):
    df_train = read_file(TRAIN_FILE,nrows)
    df_train = df_train.sample(frac=1,random_state=SEED)
    df_train,df_test = train_test_split(df_train, test_size=test_size, random_state=SEED)
    df_test,df_val = train_test_split(df_test, test_size=val_size, random_state=SEED)
    return df_train,df_val,df_test


def read_test(nrows=None):
    return read_file(TEST_FILE,nrows)


def main():
    preprocess_files()


if __name__ == "__main__":
    main()