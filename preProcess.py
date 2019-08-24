import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

def word2vec(dataset):
    corpus = []
    for i in range(dataset.shape[0]):
        review = re.sub('[^a-zA-Z]', ' ', dataset['CONTENT'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)

    cv = CountVectorizer(max_features = 1500)
    return cv.fit_transform(corpus).toarray()