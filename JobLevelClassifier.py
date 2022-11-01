
'''
Created on Jun 17, 2021

@author: Neil Kakhandiki
'''

import pandas as pd
import string
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import en_core_web_md

'''
Individual contributor: 1675
Manager: 916
Director: 424
C-level: 351
Vice President: 273
Others: 8
'''

df = pd.read_excel('Training & Predicting Raw Data.xlsx') # gets data frame 
del df['Id'] # deletes the ID column
del df['JobFunction'] # deletes the Job Function column
df.dropna(how='all', inplace=True) # removes row if each value is NaN
df['Role'] = df['Role'].fillna('') # replaces NaN values with empty string
predictDf = df[df['Role'] == ''].reset_index() # gets the predict set
trainTestDf = df[df['Role'] != ''].reset_index() # get the train/test set
del predictDf['index'] # removes unnecessary column
del trainTestDf['index'] # removes unnecessary column
trainTestDf.drop_duplicates(subset=['Title'], keep='first') # remove duplicate titles
trainTestDf['Role'] = trainTestDf['Role'].str.strip() # remove leading and trailing white space

# removes all instances of these labels
def removeLabels(labels, frame, category):
    for l in labels:
        frame.drop(index=frame[frame[category] == l].index, inplace=True)
        
removeLabels(['Others'], trainTestDf, 'Role')

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

# iterates through the train/test data frame, counts words and removes stop-words
for index, row in trainTestDf.iterrows():
    newStr = removeStopWords(str(trainTestDf.loc[index, 'Title']).lower())
    countWords(newStr)
    trainTestDf.at[index, 'Title'] = newStr
    
# converts text into a list of tokens
text_to_nlp = en_core_web_md.load()
def tokenize(text):
    clean_tokens = []
    for token in text_to_nlp(text):
        if (not token.is_stop) & (token.lemma_ != '-PRON-') & (not token.is_punct): 
            clean_tokens.append(token.lemma_)
    return clean_tokens

# x data and y data
X_text = trainTestDf['Title']
y = trainTestDf['Role']

# transform data into bag of words vectors
bow_transformer = CountVectorizer(analyzer=tokenize, max_features=len(wordCounter)).fit(X_text)
X = bow_transformer.transform(X_text)

underSample = RandomUnderSampler()
X, y = underSample.fit_resample(X, y)
print(y.value_counts())

# get training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# create, fit, and predict using Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

# get accuracy
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy of the Logistic Regression model is " + str(accuracy * 100) + "%")

def logPredict(title):
    return logistic_model.predict(bow_transformer.transform([title]))[0]

# create, fit, and predict using Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

# get accuracy
nb_accuracy = accuracy_score(y_test, nb_preds)
print("The accuracy of the Naive Bayes model is " + str(nb_accuracy * 100) + "%")

def nbPredict(title):
    return nb_model.predict(bow_transformer.transform([title]))[0]

# get the predictions with both model from the predictDf
logPreds = []
nbPreds = []
for index, row in predictDf.iterrows():
    title = str(predictDf.at[index, 'Title'])
    logPred = logPredict(title)
    nbPred = nbPredict(title)
    logPreds.append([title, logPred])
    nbPreds.append([title, nbPred])
        
naiveBayesRolePreds = pd.DataFrame(nbPreds, columns=['Title', 'Role'])
logisticRolePreds = pd.DataFrame(logPreds, columns=['Title', 'Role'])
print(logisticRolePreds.Role[:100])

# turn into excel sheets
# naiveBayesRolePreds.to_excel('naiveBayesRolePreds.xlsx')
# logisticRolePreds.to_excel('logisticRolePreds.xlsx')

