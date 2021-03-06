import random
from siamese_models import *
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,Callback
from keras import backend as K
import pandas as pd
from sklearn.metrics import log_loss
import glob


def train_model(model,df_train,df_val,epochs,prefix,early_stopping_patience,reduce_lr_patience):
    print (model.params)
    with open('models/{}.params'.format(prefix),'w') as f:
        f.write(str(model.params))

    model.model.compile(optimizer=Adam(lr=model.params.lr,clipnorm=model.params.clipnorm),loss='binary_crossentropy',
                        metrics=['binary_crossentropy'])

    train_gen = model.inputs_generator(df_train,model.params.batch_size)
    val_gen = model.inputs_generator(df_val,model.params.batch_size)
    train_steps = model.num_of_steps(df_train,model.params.batch_size)
    val_steps = model.num_of_steps(df_val,model.params.batch_size)

    callbacks = list()
    callbacks.append(ModelCheckpoint(filepath='models/{}.weights'.format(prefix),
                                    verbose=1, save_best_only=True,save_weights_only=True))
    callbacks.append(EarlyStopping(patience=early_stopping_patience, verbose=1))
    callbacks.append(ReduceLROnPlateau(factor=0.1, patience=reduce_lr_patience, verbose=1))

    history = model.model.fit_generator(train_gen,train_steps,validation_data=val_gen,
                              validation_steps=val_steps,epochs=epochs,callbacks=callbacks)
    pd.DataFrame(history.history).to_csv('models/{}.history'.format(prefix),index=False)


class LearningRateDecay(Callback):
    def __init__(self,decay):
        self.decay = decay

    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * self.decay)
        K.set_value(optimizer.lr, lr)


def params_search(experiments,epochs,nrows,early_stopping_patience,reduce_lr_patience):
    prefix = 0
    for f in glob.glob(('models/*')):
        prefix = int(f.split('/')[-1].split('.')[0])

    tokenizer1 = load(TOKENIZER_20K_1K)
    tokenizer2 = load(TOKENIZER_20K_ONE)

    for experiment in range(experiments):
        try:
            prefix += 1
            print ('Experiment:',experiment)
            tokenizer = random.choice([tokenizer1, tokenizer2])
            model = RNNSiamese(tokenizer,random_params=True)
            df_train,df_val,_ = read_train(nrows=nrows)
            train_model(model,df_train,df_val,epochs,prefix,early_stopping_patience,reduce_lr_patience)
            print ('\n\n')
        except Exception as e:
            print (e)


def evaluate(model):
    _,_,df_test = read_train()
    y_pred = model.predict(df_test)
    print (log_loss(df_test.is_duplicate,y_pred))


def main():
    tokenizer = load(TOKENIZER_20K_ONE)
    model = RNNSiamese(tokenizer)
    df_train,df_val,_ = read_train()
    train_model(model,df_train,df_val,50,'lstm_embeddings2',2,0)

    # params_search(100,50,50000,2,1)



if __name__ == "__main__":
    main()