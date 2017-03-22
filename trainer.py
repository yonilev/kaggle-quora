from models import *

tokenizer = load('tokenizer_20k_10k.p')
model = LSTMSiamese(tokenizer,Params())
model.model.compile(optimizer='adam',loss='binary_crossentropy')
df_train,df_val,df_test = read_train(nrows=1000)
train_gen = model.inputs_generator(df_train,model.params.batch_size)
val_gen = model.inputs_generator(df_val,model.params.batch_size)
train_steps = len(df_train)/model.params.batch_size
val_steps = max(len(df_val)/model.params.batch_size,1)
model.model.fit_generator(train_gen,train_steps,validation_data=val_gen,
                          validation_steps=val_steps,epochs=10)