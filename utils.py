import pickle

def save(tokenizer,file_name):
    pickle.dump( tokenizer, open( file_name, 'wb' ) )

def load(file_name):
    return pickle.load( open( file_name, 'rb' ) )