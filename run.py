from submission import *


def main():
    prefix = 'lstm_features_20k1_word2vec'
    tokenizer = load(TOKENIZER_20K_ONE)
    model = RNNSiamese(tokenizer, verbose=True)
    df_train,df_val,_ = read_train()
    train_model(model,df_train,df_val,50,prefix,1,0)
    make_submission(prefix, TOKENIZER_20K_1K)


if __name__ == "__main__":
    main()

