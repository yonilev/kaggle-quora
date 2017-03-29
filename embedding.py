from hash_tokenizer import *

PATH_FORMAT = 'embeddings/glove.6B.{}d.txt'


def embeddings_for_tokenizer(tokenizer,dim,std=0.05):
    embeddings = np.random.normal(0,std,(tokenizer.get_input_dim(),dim))
    counts = np.zeros((tokenizer.get_input_dim(),1))
    for l in open(PATH_FORMAT.format(dim)):
        spl = l.strip().split()
        word = spl[0]
        if word in tokenizer.word_counts:
            ind = tokenizer.get_word_index(word)
            embeddings[ind,:] = np.array(spl[1:])
            counts[ind] += 1

    counts[counts==0] = 1
    embeddings /= counts
    return embeddings


def main():
    tokenizer = load(TOKENIZER_ALL)
    embeddings = embeddings_for_tokenizer(tokenizer, 100)




if __name__ == "__main__":
    main()
