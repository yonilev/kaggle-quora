from training import *


def make_submission(prefix,tokenizer_file,evaluate_model=True):
    model = load_from_file(prefix,tokenizer_file)
    if evaluate_model:
        evaluate(model)

    df = read_test()
    df['is_duplicate'] = pd.Series(model.predict(df))
    df[['test_id','is_duplicate']].to_csv('submissions/{}.csv'.format(prefix),index=False)


def main():
    make_submission('lstm_embeddings2', TOKENIZER_ALL)


if __name__ == "__main__":
    main()