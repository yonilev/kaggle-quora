from submission import *


def main():
    prefix = 'lstm_all_nontrainable'
    tokenizer_name = TOKENIZER_ALL
    tokenizer = load(tokenizer_name)
    model = RNNSiamese(tokenizer, verbose=True)
    df_train,df_val,_ = read_train()
    train_model(model,df_train,df_val,50,prefix,1,0)
    make_submission(prefix, tokenizer_name)


if __name__ == "__main__":
    main()

