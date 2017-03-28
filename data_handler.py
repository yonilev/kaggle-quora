import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 1234
TRAIN_FILE = 'data/train.csv'
TEST_FILE = 'data/test.csv'


def read_file(file_name,nrows=None):
    df =  pd.read_csv(file_name,nrows=nrows)
    df['question1'] = df.question1.apply(str)
    df['question2'] = df.question2.apply(str)
    return df


def read_train(test_size=0.1,val_size=0.5,nrows=None):
    df_train = read_file(TRAIN_FILE,nrows)
    df_train = df_train.sample(frac=1)
    df_train,df_test = train_test_split(df_train, test_size=test_size, random_state=SEED)
    df_test,df_val = train_test_split(df_test, test_size=val_size, random_state=SEED)
    return df_train,df_val,df_test


def read_test(nrows=None):
    return read_file(TEST_FILE,nrows)

