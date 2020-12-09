# Text classifier
The purpose of this program is to classify text in python progrmming language using three following methods:
- Naive Bayes classifier
- Decision Tree
- KNN

Procedure of this software consist of the below phases of word processing:
## Preprocessing
In this part, different operation like:
- tokenizing
- deleting punctuations
- removing stop words
- stemming
would be performed on the input texts.
## Feature generation
Using N-grams (Unigram, Bigram and Trigram) methodologies to produce features vactor.
## Weighting features
In order to weigh feature we implement TF-IDF concepts which make it easier for accurate classification in further processing. 
## Evaluating the results
In the final stage the software would evaluate classifier using N-Gram model and result for three important criterion would be calculated which are:
- Precision
- Recall
- F1-Score