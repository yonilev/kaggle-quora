from hash_tokenizer import *

QUESTION_WORDS = 'features/question_words.txt'


class ContainsWordExtractor:
    def __init__(self,word):
        self.word = word

    def extract(self,q):
        return int(self.word in q)


class PairwiseExtractor:
    def __init__(self,extractor):
        self.extractor = extractor


class BothContainExtractor(PairwiseExtractor):
    def extract(self,q1,q2):
        return self.extractor.extract(q1) * self.extractor.extract(q2)


class OneContainsExtractor(PairwiseExtractor):
    def extract(self,q1,q2):
        e1 = self.extractor.extract(q1)
        e2 = self.extractor.extract(q2)
        if e1*e2==0 and e1+e2==1:
            return 1
        return 0


def extract_features(df):
    extractors = []

    for w in open(QUESTION_WORDS):
        w = w.strip()
        extractor = ContainsWordExtractor(w)
        extractors.append(BothContainExtractor(extractor))
        extractors.append(OneContainsExtractor(extractor))

    features = []
    for q1,q2 in zip(df.question1,df.question2):
        x = []
        q1 = q1.split()
        q2 = q2.split()
        for e in extractors:
            x.append(e.extract(q1,q2))
        features.append(x)

    return features




def main():
    df_train,_,_ = read_train(nrows=10)
    print (df_train[['question1','question2']])
    print (extract_features(df_train))

if __name__ == "__main__":
    main()