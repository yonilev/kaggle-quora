from hash_tokenizer import *

QUESTION_WORDS = 'features/question_words.txt'


######## EXTRACTORS ########

class ContainsWord:
    def __init__(self,word):
        self.word = word

    def extract(self,q):
        return int(self.word in q)


class NumberOfWords:
    def extract(self,q):
        return len(q)


class UniqueWords:
    def extract(self,q):
        return set(q)


######## FEATURES ##########

class PairwiseFeature:
    def __init__(self,extractor):
        self.extractor = extractor

    def extract_from(self,q1,q2):
        return self.extractor.extract(q1),self.extractor.extract(q2)


class BothContainFeature(PairwiseFeature):
    def extract(self,q1,q2):
        e1,e2 = self.extract_from(q1,q2)
        return e1 * e2


class OneContainsFeature(PairwiseFeature):
    def extract(self,q1,q2):
        e1,e2 = self.extract_from(q1,q2)
        if e1*e2==0 and e1+e2==1:
            return 1
        return 0


class RatioFeature(PairwiseFeature):
    def extract(self,q1,q2):
        e1,e2 = self.extract_from(q1,q2)
        return min(e1,e2)/max(e1,e2)


class JaccardFeature(PairwiseFeature):
    def extract(self,q1,q2):
        e1,e2 = self.extract_from(q1,q2)
        return len(e1&e2)/len(e1|e2)


class JaccardIdfFeature(PairwiseFeature):
    def __init__(self,extractor,tokenizer):
        super(JaccardIdfFeature, self).__init__(extractor)
        self.tokenizer = tokenizer

    def compute_idf(self,w):
        df = self.tokenizer.word_docs.get(w,0)
        return np.log((self.tokenizer.document_count)/(df+1))

    def extract(self,q1,q2):
        e1,e2 = self.extract_from(q1,q2)
        n = 0
        d = 0
        for w in e1&e2:
            n += self.compute_idf(w)
        for w in e1|e2:
            d += self.compute_idf(w)
        return n/d


class FeaturesExtractor:
    def __init__(self,tokenizer):
        self.features = list()
        self.features.append(JaccardIdfFeature(UniqueWords(), tokenizer))
        self.features.append(JaccardFeature(UniqueWords()))
        self.features.append(RatioFeature(NumberOfWords()))

        for w in open(QUESTION_WORDS):
            w = w.strip()
            extractor = ContainsWord(w)
            self.features.append(BothContainFeature(extractor))
            self.features.append(OneContainsFeature(extractor))

    def extract(self,df):
        extracted_features = list()
        for q1,q2 in zip(df.question1,df.question2):
            q1 = text_to_word_sequence(q1)
            q2 = text_to_word_sequence(q2)
            x = []
            for f in self.features:
                x.append(f.extract(q1,q2))
            extracted_features.append(x)

        return extracted_features

    def number_of_features(self):
        return len(self.features)


def main():
    pass

if __name__ == "__main__":
    main()