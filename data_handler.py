import pandas as pd

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

def read_file(file_name):
    df =  pd.read_csv(file_name)
    df['question1'] = df.question1.apply(str)
    df['question2'] = df.question2.apply(str)
    return df


def read_train():
    return read_file(TRAIN_FILE)


def read_test():
    return read_file(TEST_FILE)