from tokenizer import Tokenize
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics



obj = Tokenize()
obj.retToken()
aa = pd.DataFrame(obj.dictionary.items(), columns=['sms','label'])

uni_gram = CountVectorizer(ngram_range=(1, 1))
bi_gram = CountVectorizer(ngram_range=(1, 2))
tri_gram = CountVectorizer(ngram_range=(1, 3))

#count_vect = CountVectorizer()
#print(type(aa['sms']))

X_train_counts_one = uni_gram.fit_transform(aa['sms'])
X_train_counts_two = bi_gram.fit_transform(aa['sms'])
X_train_counts_three = tri_gram.fit_transform(aa['sms'])
#print(count_vect.vocabulary_)

tfidf_transformer = TfidfTransformer()

X_train_tfidf_one = tfidf_transformer.fit_transform(X_train_counts_one)
X_train_tfidf_two = tfidf_transformer.fit_transform(X_train_counts_two)
X_train_tfidf_three = tfidf_transformer.fit_transform(X_train_counts_three)
#print(X_train_tfidf)

X_train_one, X_test_one, Y_train_one, Y_test_one = train_test_split(X_train_tfidf_one, aa['label'].values, test_size=0.25, random_state=1000)
X_train_two, X_test_two, Y_train_two, Y_test_two = train_test_split(X_train_tfidf_two, aa['label'].values, test_size=0.25, random_state=1000)
X_train_three, X_test_three, Y_train_three, Y_test_three = train_test_split(X_train_tfidf_three, aa['label'].values, test_size=0.25, random_state=1000)

clf_one = MultinomialNB()
clf_one.fit(X_train_one, Y_train_one)
predicted_one = clf_one.predict(X_test_one)

clf_two = MultinomialNB()
clf_two.fit(X_train_two, Y_train_two)
predicted_two = clf_two.predict(X_test_two)

clf_three = MultinomialNB()
clf_three.fit(X_train_three, Y_train_three)
predicted_three = clf_three.predict(X_test_three)

#print(list(predicted))

print(aa.head())
print('for the unigram we have:')
print(metrics.classification_report(Y_test_one, predicted_one,target_names=['ham','spam']))

print('for the bigram we have:')
print(metrics.classification_report(Y_test_two, predicted_two,target_names=['ham','spam']))

print('for the trigram we have:')
print(metrics.classification_report(Y_test_three, predicted_three,target_names=['ham','spam']))


query = [input('input the qurey:')]
q_count = CountVectorizer()
q_train = tri_gram.transform(query)
q_tfIdf = tfidf_transformer.transform(q_train)
predicted_query = clf_three.predict(q_tfIdf)
print("the predicted value for the query is:" , predicted_query)
