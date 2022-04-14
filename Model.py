import pickle
import re


from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

modelscorev2 = joblib.load('model.pkl' , mmap_mode ='r')
print(modelscorev2)
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()
document = ["Đây là ly quán giao mình mình đã bị 2 lần như thế"]
count_vect.fit_transform(document)
X_new_counts = count_vect.fit_transform(document)
# We call transform instead of fit_transform because it's already been fit
X_new_tfidf = tfidf_transformer.fit_transform(X_new_counts)
print(modelscorev2.predict(document))