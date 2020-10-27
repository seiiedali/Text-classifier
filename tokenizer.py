import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


class Tokenize:
    def __init__(self):
        self.tokens = dict()
        self.allsms = []
        self.dictionary = dict()


    def remove_punctuation(self,s):
        punctuation = "0123456789!()-[]{};:'\"\,<>./?@#$%^&*_~"
        no_punct = ""
        for c in s:
            if c not in punctuation:
                no_punct += c
        return no_punct

    def retToken(self):
        counter =0
        path = "./Docs/" + "SMSSpamCollection.txt"
        with open(path, "r" , encoding='UTF-8') as content:
            content = self.remove_punctuation(content.read())
            content = content.splitlines()
            for sms in content:
                tokens = re.split(r'\t|\s', sms.lower())
                filtered_words = [word for word in tokens if word not in stopwords.words('english')]

                ps = PorterStemmer()
                stem = [ps.stem(token) for token in filtered_words]
                #print(stem)

                #lem = WordNetLemmatizer()
                #lemit = [lem.lemmatize(token, "v") for token in stem]

                counter += 1
                self.tokens[counter] = stem

                str = ""
                for key in tokens:
                    tempStr = tokens[1:]
                    str = " ".join(tempStr)
                    self.allsms.append(str)

                self.dictionary[str] = stem[0]