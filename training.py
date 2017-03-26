from models import *
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import pandas as pd
from sklearn.metrics import log_loss


def train_model(model,df_train,df_val,epochs,prefix,early_stopping_patience=4,reduce_lr_patience=2):
    print model.params
    with open('models/{}.params'.format(prefix),'w') as f:
        f.write(str(model.params))

    model.model.compile(optimizer=Adam(model.params.lr),loss='binary_crossentropy',
                        metrics=['binary_crossentropy'])

    train_gen = model.inputs_generator(df_train,model.params.batch_size)
    val_gen = model.inputs_generator(df_val,model.params.batch_size)
    train_steps = model.num_of_steps(df_train,model.params.batch_size)
    val_steps = model.num_of_steps(df_val,model.params.batch_size)

    callbacks = list()
    callbacks.append(ModelCheckpoint(filepath='models/'+prefix+'.weights',
                                     monitor='val_binary_crossentropy', verbose=1,
                                     save_best_only=True,save_weights_only=True))
    callbacks.append(EarlyStopping(monitor='val_binary_crossentropy', patience=early_stopping_patience, verbose=1))
    callbacks.append(ReduceLROnPlateau(monitor='val_binary_crossentropy', factor=0.1, patience=reduce_lr_patience))

    history = model.model.fit_generator(train_gen,train_steps,validation_data=val_gen,
                              validation_steps=val_steps,epochs=epochs,callbacks=callbacks)
    pd.DataFrame(history.history).to_csv('models/{}.history'.format(prefix),index=False)


def evaluate(model):
    _,_,df_test = read_train()
    y_pred = model.predict(df_test)
    print log_loss(df_test.is_duplicate,y_pred)


def main():
    tokenizer = load(TOKENIZER_FILE)
    model = CNNSiamese(tokenizer,Params())
    df_train,df_val,df_test = read_train()
    train_model(model,df_train,df_val,40,MODEL_PREFIX)
    # model = load_from_file(MODEL_PREFIX,TOKENIZER_FILE,LSTMSiamese)
    # evaluate(model)



if __name__ == "__main__":
    main()