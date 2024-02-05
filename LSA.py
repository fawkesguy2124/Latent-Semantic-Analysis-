import os
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

documents_list = []
with open( os.path.join("articles.txt") ,"r") as fin:
    for line in fin.readlines():
        text = line.strip()
        documents_list.append(text)

tokenizer = RegexpTokenizer(r'\w+')


tfidf = TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,1),
                        tokenizer = tokenizer.tokenize)


train_data = tfidf.fit_transform(documents_list)


num_components=10


lsa = TruncatedSVD(n_components=num_components, n_iter=100, random_state=42)


lsa.fit_transform(train_data)


Sigma = lsa.singular_values_
V_transpose = lsa.components_.T



terms = tfidf.get_feature_names_out()

for index, component in enumerate(lsa.components_):
    zipped = zip(terms, component)
    top_terms_key=sorted(zipped, key = lambda t: t[1], reverse=True)[:5]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(index)+": ",top_terms_list)
