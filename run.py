from submission import *


def main():
    prefix = 'lstm'
    tokenizer = load(TOKENIZER_20K_1K)
    model = LSTMSiamese(tokenizer)
    df_train,df_val,_ = read_train()
    train_model(model,df_train,df_val,50,prefix,2,0)
    make_submission(prefix, TOKENIZER_20K_1K)


if __name__ == "__main__":
    main()

