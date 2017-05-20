from hash_tokenizer import *

GLOVE = 'embeddings/glove.42B.300d.txt'
GLOVE_PROCESSED = 'embeddings/glove_processed.42.300d.txt'
DIM = 300


def preprocess(tokenizer):
    with open(GLOVE_PROCESSED,'w') as f:
        for l in open(GLOVE):
            spl = l.strip().split()
            if len(spl)!=DIM+1:
                continue
            word = spl[0].lower()
            if not tokenizer.is_unknown(word):
                f.write(l)


def embeddings_for_tokenizer(tokenizer,std=0.05):
    embeddings = np.zeros((tokenizer.get_input_dim(),DIM))
    counts = [0] * tokenizer.get_input_dim()
    for l in open(GLOVE_PROCESSED):
        spl = l.strip().split()
        word = spl[0].lower()
        ind = tokenizer.get_word_index(word)
        embeddings[ind,:] += np.array([float(x) for x in spl[1:]])
        counts[ind] += 1

    t = 0
    for i,c in enumerate(counts):
        if c==0:
            embeddings[i] = np.random.normal(scale=std,size=DIM)
        else:
            embeddings[i] /= c
            t+=1
    print('embeddings coverage:', t / len(counts))

    # for word in tokenizer.word_index:
    #     ind = tokenizer.get_word_index(word)
    #     if ind<len(counts) and counts[ind]==0:
    #         print (word)

    return embeddings


def main():
    tokenizer = load(TOKENIZER_ALL)
    preprocess(tokenizer)
    embeddings_for_tokenizer(tokenizer)


if __name__ == "__main__":
    main()
