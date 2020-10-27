from tokenizer import Tokenize
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

class Ngram:

    def unigram(allsms):
        uni_gram = TfidfVectorizer(ngram_range=(1, 1))
        x1 = uni_gram.fit_transform(allsms)
        return x1

    def bigram(allsms):
        bi_gram = CountVectorizer(ngram_range=(2, 2))
        x2 = bi_gram.fit_transform(allsms)
        print(bi_gram.get_feature_names())

    def trigram(allsms):
        tri_gram = CountVectorizer(ngram_range=(3, 3))
        x3 = tri_gram.fit_transform(allsms)
        print(tri_gram.get_feature_names())