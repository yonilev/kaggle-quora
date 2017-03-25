from training import *


def make_submission(prefix,tokenizer_file,model_class):
    model = load_from_file(prefix,tokenizer_file,model_class)
    df = read_test()
    df['is_duplicate'] = pd.Series(model.predict(df))
    df[['test_id','is_duplicate']].to_csv('submissions/{}.csv'.format(prefix),index=False)


def main():
    make_submission(MODEL_PREFIX,TOKENIZER_FILE,LSTMSiamese)


if __name__ == "__main__":
    main()