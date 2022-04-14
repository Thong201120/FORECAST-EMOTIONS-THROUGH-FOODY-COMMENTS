from gensim.models import Word2Vec
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# path = './CSVData'


# def readdata(path):
#     list_file = os.listdir(path)
#     data = pd.DataFrame()
#     for filename in list_file:
#         data = pd.concat([data, pd.read_csv(os.path.join(path, filename), sep=',')])
#
#     return data.Review, data.Label
#
#
# reviews, labels = readdata(path)
#
# print(reviews)
# print(labels)
#
# input_gensim = []
# for review in reviews:
#     input_gensim.append(review.split())
#
# model = Word2Vec(input_gensim, size=128, window=5, min_count=0, workers=4, sg=1)
# model.wv.save("word.model")
#
# import gensim.models.keyedvectors as word2vec
#
# model_embedding = word2vec.KeyedVectors.load('./word.model')
#
# word_labels = []
# max_seq = 200
# embedding_size = 128
#
# for word in model_embedding.vocab.keys():
#     word_labels.append(word)
#
#
# def comment_embedding(comment):
#     matrix = np.zeros((max_seq, embedding_size))
#     words = comment.split()
#     lencmt = len(words)
#
#     for i in range(max_seq):
#         indexword = i % lencmt
#         if (max_seq - i < lencmt):
#             break
#         if (words[indexword] in word_labels):
#             matrix[i] = model_embedding[words[indexword]]
#     matrix = np.array(matrix)
#     return matrix
#
# train_data = []
# label_data = []
#
# for x in tqdm(reviews):
#     train_data.append(comment_embedding(x))
# train_data = np.array(train_data)
#
# for y in tqdm(labels):
#     label_ = np.zeros(3)
#     try:
#         label_[int(y)] = 1
#     except:
#         label_[0] = 1
#     label_data.append(label_)
#
# from tensorflow.keras import layers
# from tensorflow import keras
# import tensorflow as tf
# from keras.preprocessing import sequence
#
# sequence_length = 200
# embedding_size = 128
# num_classes = 3
# filter_sizes = 3
# num_filters = 150
# epochs = 50
# batch_size = 30
# learning_rate = 0.01
# dropout_rate = 0.5
#
# x_train = train_data.reshape(train_data.shape[0], sequence_length, embedding_size, 1).astype('float32')
# y_train = np.array(label_data)
#
# # Define model
# model = keras.Sequential()
# model.add(layers.Convolution2D(num_filters, (filter_sizes, embedding_size),
#                         padding='valid',
#                         input_shape=(sequence_length, embedding_size, 1), activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(198, 1)))
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(3, activation='softmax'))
# # Train model
# adam = tf.train.AdamOptimizer()
# model.compile(loss='categorical_crossentropy',
#               optimizer=adam,
#               metrics=['accuracy'])
# print(model.summary())