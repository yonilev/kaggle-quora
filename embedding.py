from hash_tokenizer import *

PATH_FORMAT = 'embeddings/glove.6B.{}d.txt'


def embeddings_for_tokenizer(tokenizer,dim,std=0.05):
    embeddings = np.zeros((tokenizer.get_input_dim(),dim))
    counts = [0] * tokenizer.get_input_dim()
    for l in open(PATH_FORMAT.format(dim)):
        spl = l.strip().split()
        word = spl[0]
        if word in tokenizer.word_counts:
            ind = tokenizer.get_word_index(word)
            embeddings[ind,:] += np.array([float(x) for x in spl[1:]])
            counts[ind] += 1

    found = 0
    for c in counts:
        if c>0:
            found+=1
    print('found: {}'.format(found/len(counts)))

    for i,c in enumerate(counts):
        if c==0:
            embeddings[i] = np.random.normal(scale=std,size=dim)
        else:
            embeddings[i] /= c

    return embeddings


def main():
    tokenizer = load(TOKENIZER_20K_10K)
    embeddings = embeddings_for_tokenizer(tokenizer, 100)


if __name__ == "__main__":
    main()
