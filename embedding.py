from hash_tokenizer import *
from gensim.models.word2vec import Word2Vec

GLOVE_FORMAT = 'embeddings/glove.6B.{}d.txt'
WORD2VEC_FORMAT = 'embeddings/word2vec.traintest.{}d.txt'



def embeddings_for_tokenizer(tokenizer,dim,std=0.05):
    embeddings = np.zeros((tokenizer.get_input_dim(),dim))
    counts = [0] * tokenizer.get_input_dim()
    for l in open(WORD2VEC_FORMAT.format(dim)):
        spl = l.strip().split()
        word = spl[0]
        if word in tokenizer.word_counts:
            ind = tokenizer.get_word_index(word)
            embeddings[ind,:] += np.array([float(x) for x in spl[1:]])
            counts[ind] += 1

    t = 0
    for c in counts:
        if c>0:
            t+=1
    print ('embeddings coverage:',t/len(counts))

    for i,c in enumerate(counts):
        if c==0:
            embeddings[i] = np.random.normal(scale=std,size=dim)
        else:
            embeddings[i] /= c

    return embeddings


def train_embedding(dim):
    model = Word2Vec(MySentences(),size=dim)
    word_vectors = model.wv
    del model
    with open(WORD2VEC_FORMAT.format(dim),'w') as f:
        for w in word_vectors.vocab:
            v = word_vectors[w].tolist()
            v = [w] + [str(x) for x in v]
            f.write(' '.join(v) + '\n')


class MySentences(object):
    def __iter__(self):
        for df in pd.read_csv(preprocessed_name(TRAIN_FILE), chunksize=10000):
            for sent in np.concatenate([df.question1,df.question2]):
                yield sent.split()
        for df in pd.read_csv(preprocessed_name(TEST_FILE), chunksize=10000):
            for sent in np.concatenate([df.question1,df.question2]):
                yield sent.split()


def main():
    # tokenizer = load(TOKENIZER_20K_10K)
    # embeddings = embeddings_for_tokenizer(tokenizer, 100)
    train_embedding(300)



if __name__ == "__main__":
    main()
