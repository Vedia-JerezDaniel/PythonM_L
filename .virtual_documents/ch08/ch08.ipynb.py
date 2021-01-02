import os
import sys
import tarfile
import time
import urllib.request


source = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
target = 'aclImdb_v1.tar.gz'

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = progress_size / (1024.**2 * duration)
    percent = count * block_size * 100. / total_size

    sys.stdout.write("\rget_ipython().run_line_magic("d%%", " | %d MB | %.2f MB/s | %d sec elapsed\" %")
                    (percent, progress_size / (1024.**2), speed, duration))
    sys.stdout.flush()


if not os.path.isdir('aclImdb') and not os.path.isfile('aclImdb_v1.tar.gz'):
    urllib.request.urlretrieve(source, target, reporthook)


if not os.path.isdir('aclImdb'):

    with tarfile.open(target, 'r:gz') as tar:
        tar.extractall()


import pyprind
import pandas as pd
import os

# change the `basepath` to the directory of the
# unzipped movie dataset

basepath = 'aclImdb'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']


df.head()


import numpy as np # to permute the DB

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))


df.to_csv('movie_data.csv', index=False, encoding='utf-8')


df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(5)


df.shape


from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'Everything in life is hard',
        'but someones are easy',
        'The sun is shining, the weather is sweet, and one and one is two'])

bag = count.fit_transform(docs)


print(count.vocabulary_)


print(bag.toarray())


np.set_printoptions(precision=2)


from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True, 
                         norm='l2', smooth_idf=True)

print(tfidf.fit_transform(count.fit_transform(docs)).toarray())


tf_is = 3
n_docs = 5
idf_is = np.log((n_docs+1) / (3+1))
tfidf_is = tf_is * (idf_is + 1)
print('tf-idf of term "is" = get_ipython().run_line_magic(".2f'", " % tfidf_is)")


tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
raw_tfidf


l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))
l2_tfidf


df.loc[0, 'review'][-50:]


import re

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


preprocessor(df.loc[0, 'review'][-50:])


preprocessor("</a>This :) is :( a test :-)get_ipython().getoutput("")")


df['review'] = df['review'].apply(preprocessor)


from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


tokenizer('runners like running and thus they run')


tokenizer_porter('runners like running and thus they run')


import nltk

nltk.download('stopwords')


from nltk.corpus import stopwords

stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
if w not in stop]


X_train = df.loc[:500, 'review'].values
y_train = df.loc[:500, 'sentiment'].values
X_test = df.loc[1500:, 'review'].values
y_test = df.loc[1500:, 'sentiment'].values


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False, preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0, solver='liblinear'))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5, verbose=2,
                           n_jobs=4)


gs_lr_tfidf.fit(X_train, y_train)


print('Best parameter set: get_ipython().run_line_magic("s", " ' % gs_lr_tfidf.best_params_)")
print('CV Accuracy: get_ipython().run_line_magic(".3f'", " % gs_lr_tfidf.best_score_)")


clf = gs_lr_tfidf.best_estimator_

print('Test Accuracy: get_ipython().run_line_magic(".3f'", " % clf.score(X_test, y_test))")


from sklearn.linear_model import LogisticRegression
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

np.random.seed(0)
np.set_printoptions(precision=6)
y = [np.random.randint(3) for i in range(25)]
X = (y + np.random.randn(25)).reshape(-1, 1)

cv5_idx = list(StratifiedKFold(n_splits=5, shuffle=False).split(X, y))
    
lr = LogisticRegression(random_state=123, multi_class='ovr', solver='lbfgs', n_jobs=4)

cross_val_score(lr, X, y, cv=cv5_idx)


from sklearn.model_selection import GridSearchCV

lr = LogisticRegression(solver='lbfgs', multi_class='ovr', random_state=1, n_jobs=4)
gs = GridSearchCV(lr, {}, cv=cv5_idx, verbose=3, n_jobs=4).fit(X, y) 


gs.best_score_


lr = LogisticRegression(solver='lbfgs', multi_class='ovr', random_state=1)
cross_val_score(lr, X, y, cv=cv5_idx).mean()


# This cell is not contained in the book but
# added for convenience so that the notebook
# can be executed starting here, without
# executing prior code in this notebook

import os
import gzip

if not os.path.isfile('movie_data.csv'):
    if not os.path.isfile('movie_data.csv.gz'):
        print('Please place a copy of the movie_data.csv.gz'
              'in this directory. You can obtain it by'
              'a) executing the code in the beginning of this'
              'notebook or b) by downloading it from GitHub:'
              'https://github.com/rasbt/python-machine-learning-'
              'book-2nd-edition/blob/master/code/ch08/movie_data.csv.gz')
    else:
        with gzip.open('movie_data.csv.gz', 'rb') as in_f, \
                open('movie_data.csv', 'wb') as out_f:
            out_f.write(in_f.read())


import numpy as np
import re
from nltk.corpus import stopwords


# The `stop` is defined as earlier in this chapter
# Added it here for convenience, so that this section
# can be run as standalone without executing prior code
# in the directory
stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


next(stream_docs(path='movie_data.csv'))


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)


from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version

clf = SGDClassifier(loss='log', random_state=1, n_jobs=4)
doc_stream = stream_docs(path='movie_data.csv')


import pyprind
pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()


X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: get_ipython().run_line_magic(".3f'", " % clf.score(X_test, y_test))")


clf = clf.partial_fit(X_test, y_test)



mv = pd.read_csv('movie_data.csv', encoding='utf-8')
mv.head(6)


from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english',
                        max_df=.1,
                        max_features=5000)
X = count.fit_transform(mv['review'].values)


from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch', n_jobs=4)
X_topics = lda.fit_transform(X)


lda.components_.shape


n_top_words = 7
feature_names = count.get_feature_names()

for idx, topic in enumerate(lda.components_):
    print("Topic get_ipython().run_line_magic("d:"", " % (idx + 1))")
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words -1 :-1]]))


horror = X_topics[:, 5].argsort()[::-1]

for idx, movie_idx in enumerate(horror[:3]):
    print('\nHorror movie #get_ipython().run_line_magic("d:'", " % (idx + 1))")
    print(df['review'][movie_idx][:300], '...')


! python ../.convert_notebook_to_script.py --input ch08.ipynb --output ch08.py
