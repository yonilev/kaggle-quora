from training import *


def make_submission(prefix,tokenizer_file,model_class,evaluate_model=True):
    model = load_from_file(prefix,tokenizer_file,model_class)
    if evaluate_model:
        evaluate(model)

    df = read_test()
    df['is_duplicate'] = pd.Series(model.predict(df))
    df[['test_id','is_duplicate']].to_csv('submissions/{}.csv'.format(prefix),index=False)


def main():
    make_submission('lstm_embeddings', TOKENIZER_20K_10K, LSTMSiamese)


if __name__ == "__main__":
    main()