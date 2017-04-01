import pandas as pd
from sklearn.model_selection import train_test_split
from nltk import word_tokenize


SEED = 1234
TRAIN_FILE = 'data/train.csv'
TEST_FILE = 'data/test.csv'
PREPROCESSED = '_preprocessed'


def read_file(file_name,nrows=None,preprocessed=True):
    if preprocessed:
        file_name = file_name.replace('.csv', '%s.csv' % PREPROCESSED)
    df =  pd.read_csv(file_name,nrows=nrows)
    df['question1'] = df.question1.apply(str)
    df['question2'] = df.question2.apply(str)

    return df


def preprocess_files():
    preprocess_file(TRAIN_FILE)
    preprocess_file(TEST_FILE)


def preprocess_file(file):
    df = read_file(file,preprocessed=False)
    df['question1'] = df.question1.apply(tokenize_text)
    df['question2'] = df.question2.apply(tokenize_text)
    df.to_csv(file.replace('.csv', '%s.csv' % PREPROCESSED), index=False)


def tokenize_text(text):
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