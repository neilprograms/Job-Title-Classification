
'''
Created on Jun 20, 2021

@author: neil
'''

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
import seaborn as sns

df = pd.read_excel('Training & Predicting Raw Data.xlsx') # gets data frame 
del df['Id'] # deletes the ID column
del df['Role'] # deletes the Job Function column
df.dropna(how='all', inplace=True) # removes row if each value is NaN
df['JobFunction'] = df['JobFunction'].fillna('') # replaces NaN values with empty string
df.replace('Not Available', '', inplace=True) # replace NA with empty string

predictDf = df[df['JobFunction'] == ''].reset_index() # gets the predict set
trainTestDf = df[df['JobFunction'] != ''].reset_index() # get the train/test set
del predictDf['index'] # removes unnecessary column
del trainTestDf['index'] # removes unnecessary column

trainTestDf.drop_duplicates(subset=['Title'], keep='first') # remove duplicate titles
trainTestDf['JobFunction'] = trainTestDf['JobFunction'].str.strip() # remove leading and trailing white space

# removes all instances of these labels
def removeLabels(labels, frame, category):
    for l in labels:
        frame.drop(index=frame[frame[category] == l].index, inplace=True)
        
removeLabels(['Others', 'Operations', 'Finance', 'Procurement', 
              'HR', 'IT - Network', 'Unknown'], trainTestDf, 'JobFunction')

# job function labels
job_function = ["IT", "Marketing/Sales", "Network", "Security"]

stopWords = set(stopwords.words('english')) # stop-words
vocab = string.ascii_lowercase + " " # vocabulary (only alphabet + space)

# removes stop-words from a string
def removeStopWords(text):
    # only alphabet
    cleanText = ""
    for ch in text:
        if ch in vocab:
            cleanText += ch

    newStr = ""
    for word in cleanText.split(' '):
        if word not in stopWords:
            newStr += word + " ";

    return newStr.strip()

wordCounter = {} # dictionary of occurrences of words

# count occurrences of the words
def countWords(text):
    for word in text.split(' '):
        if word in wordCounter:
            wordCounter[word] = wordCounter[word] + 1
        else:
            wordCounter[word] = 1

# iterates through the data frame, counts words and removes stop-words
for index, row in trainTestDf.iterrows():
    newStr = removeStopWords(str(trainTestDf.loc[index, 'Title']).lower())
    countWords(newStr)
    trainTestDf.loc[index, 'Title'] = newStr

num_words = len(wordCounter)

# tokenize the words
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(trainTestDf['Title'])
wordToIndex = tokenizer.word_index

# pad the sequences to same length
X = tokenizer.texts_to_sequences(trainTestDf['Title'])
X = pad_sequences(X, maxlen=10, dtype='int32', padding='post', truncating='post')

print(X[:10])
print(trainTestDf['JobFunction'][:10])

# under samples the data
under = RandomUnderSampler()
X, y = under.fit_resample(X, trainTestDf['JobFunction'])

# convert labels to one hot encodings
Y = pd.get_dummies(y.values)
print(type(X), type(Y))

# split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)

# data shapes
print("X and Y data shape:", X.shape, Y.shape)
print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)
print()

# decodes a padded sequence to the word using wordToIndex
def decode(padSeq):
    newStr = ""
    for num in padSeq:
        if num != 0:
            for key, value in wordToIndex.items():
                if value == num:
                    newStr += key + " "
    return newStr.strip()

# Model from this article
# https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
# MODEL #1
'''
model = Sequential()
model.add(Embedding(num_words, 100, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, activation='softmax'))
# categorical_crossentropy because labels are one hot encoded
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print()

history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, 
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accuracy = model.evaluate(X_test, y_test)
print('Test Set Loss: {:0.3f}\nTest Set Accuracy: {:0.3f}'.format(accuracy[0], accuracy[1]))
print()
'''
# Model from this article
# https://djajafer.medium.com/multi-class-text-classification-with-keras-and-lstm-4c5525bef592
# MODEL #2

model = Sequential()
model.add(Embedding(num_words, 64))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(4, activation='softmax'))
print(model.summary())
# categorical_crossentropy because labels are one hot encoded
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

history = model.fit(X_train, y_train, epochs=20, 
                    validation_data=(X_test, y_test), verbose=2)

accuracy = model.evaluate(X_test, y_test)
print('Test Set Loss: {:0.3f}\nTest Set Accuracy: {:0.3f}'.format(accuracy[0], accuracy[1]))
print()


# MODEL #3
'''
model = Sequential()
model.add(Embedding(num_words, 32, input_length=10))
model.add(LSTM(64, dropout=0.1))
model.add(Dense(4, activation="sigmoid"))
print(model.summary())
# categorical_crossentropy because labels are one hot encoded
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test),)

accuracy = model.evaluate(X_test, y_test)
print('Test Set Loss: {:0.3f}\nTest Set Accuracy: {:0.3f}'.format(accuracy[0], accuracy[1]))
print()
'''

# plots accuracy and loss
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
  
# plot_graphs(history, "accuracy")
# plot_graphs(history, "loss")

y_pred = []
for title in X_test:
    seq = tokenizer.texts_to_sequences([decode(title)])
    padded = pad_sequences(seq, maxlen=10, dtype='int32', padding='post', truncating='post')
    pred = model.predict(padded)
    y_pred += pred.tolist()
y_pred = np.argmax(y_pred, axis=1).tolist()

y_true = y_test.values.tolist()
y_true = np.argmax(y_true, axis=1).tolist()

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
sns.heatmap(cm, annot=True, xticklabels=job_function, yticklabels=job_function) 
plt.show()

FP = (cm.sum(axis=0) - np.diag(cm)).astype(float)
FN = (cm.sum(axis=1) - np.diag(cm)).astype(float)
TP = (np.diag(cm)).astype(float)
TN = (cm.sum() - (FP + FN + TP)).astype(float)

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (FP + TP)
recall = TP / (FN + TP)
f1_score = 2 * ((precision * recall) / (precision + recall))

print("Global Accuracy:", np.diag(cm).sum() / cm.sum())
print("Accuracy:", dict(zip(job_function, accuracy.tolist())))
print("Precision:", dict(zip(job_function, precision.tolist())))
print("Recall:", dict(zip(job_function, recall.tolist())))
print("F1 Score:", dict(zip(job_function, f1_score.tolist())))

# predict the level from string
def predict(title):
    seq = tokenizer.texts_to_sequences([title])
    padded = pad_sequences(seq, maxlen=10, dtype='int32', padding='post', truncating='post')
    pred = model.predict(padded)
    return job_function[np.argmax(pred)]


# get the predictions with model from the predictDf
preds = []
for index, row in predictDf.iterrows():
    title = str(predictDf.at[index, 'Title'])
    pred = predict(title)
    preds.append([title, pred])
        
LSTMFunctionPreds = pd.DataFrame(preds, columns=['Title', 'JobFunction'])

# turn into excel sheets
LSTMFunctionPreds.to_excel('LSTMFunctionUnder.xlsx')

'''
Job Function Classifier Metrics:
Global Accuracy: 0.967
Accuracy: {'IT': 0.975, 'Marketing/Sales': 0.993, 'Network': 0.989, 'Security': 0.978}
Precision: {'IT': 0.938, 'Marketing/Sales': 0.985, 'Network': 0.971, 'Security': 0.983}
Recall: {'IT': 0.9740, 'Marketing/Sales': 0.985, 'Network': 0.985, 'Security': 0.921}
F1 Score: {'IT': 0.955, 'Marketing/Sales': 0.985, 'Network': 0.978, 'Security': 0.951}
'''
    