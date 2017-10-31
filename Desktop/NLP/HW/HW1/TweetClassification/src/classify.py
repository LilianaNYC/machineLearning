import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import re
import string
import scipy.sparse as sp
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix
import sys

def add_features(data_train):
    numEmo = [num_emoji(data_train) for data_train in data_train]
    numChar = [num_char(data_train) for data_train in data_train]
    numWord = [num_word(data_train) for data_train in data_train]
    numExp = [num_expression(data_train) for data_train in data_train]
    numRepet = [is_long(data_train) for data_train in data_train]
    numHashTags =[num_hashtags(data_train) for data_train in data_train]
    isRT =[is_RT(data_train) for data_train in data_train]
    
    punct = [is_punct(data_train) for data_train in data_train]
    upperC = [is_upperC(data_train) for data_train in data_train]
    features = np.column_stack((punct, upperC, numEmo, numChar, numWord, numExp, numRepet, numHashTags, isRT))
    return (features)

def is_upperC(text):
    for character in text:
        if text[0].isupper():
            return 1
    return 0

def is_punct(text):
    for character in text:
        if character in string.punctuation:
            return 1
    return 0

def is_long(text):
    return (len(re.findall(r'(.)\1\1+',text)))

def num_hashtags(text):
    return (text.count('HASH') + text.count('HNDL') + text.count('#') + text.count('@'))

def num_expression(text):
    return (text.count('!') + text.count('?'))

def num_word(text):
    word=1
    for i in text:
        if(i==' '):
            word=word+1
    return (word)

def num_char(text):
    char=0
    for i in text:
        char=char+1
    return (char)

def num_emoji(text):
    return (len(re.findall(u"[^\u0000-\u007e]+", text)))


def is_RT(text):
    for character in text:
        if character in re.findall('RT',text):
            return 1
    return 0

def num_URL(data):
    return (data.count('http'))

def pre_pross(data):
    data = [re.sub(r'#(\w+)', r'HASH\1', w) for w in data]
    data = [re.sub(r'@(\w+)', r'HNDL\1', w) for w in data]
    numURL = [num_URL(data) for data in data]
    #data = [item.lower() for item in data] # turns all the items into lower case
    #data = [re.sub(r'(.)\1+', r'\1\1', w) for w in data]#DOUBLE LETTERS
    #data = [w.replace("'", '') for w in data]#eliminates '
    data = [w.replace('http : //t', '') for w in data]#eliminates '
    data = [w.replace('is', '') for w in data]
    data = [w.replace('at', '') for w in data]
    data = [w.replace('which', '') for w in data]
    data = [w.replace('on', '') for w in data]
    #data = [w.split(': //t', 1)[0] for w in data]
    
    #removes endings
    regx = re.compile('(ed\\b)|(ing\\b)|(s\\b)|(en\\b)')
    def repl(mat, dic = {1:'',2:'', 3:'',4:''}):
        return dic[mat.lastindex]
    data = [regx.sub(repl,data) for data in data]
    
    return (data, numURL)

def read_file(file):
    text=[]
    party=[]
    read_file= open(file, 'r')
    split= [line.strip() for line in read_file]
    for line in split:
        text.append(line.split('\t')[0])
        party.append(line.split('\t')[1])
    return(text, party)

def vec_data(data_textTrain, data_textTest, gram1, gram2):
    vectorizer =  CountVectorizer(decode_error='strict',analyzer='word',ngram_range=(gram1,gram2), stop_words='english')
    (data_textTrain, numURLtrain) = pre_pross(data_textTrain)
    X_train = vectorizer.fit_transform(data_textTrain)
    
    feature_names = vectorizer.get_feature_names()
    
    
    featuresTrain = add_features(data_textTrain)
    featuresTrain = np.column_stack((featuresTrain, numURLtrain))
    X_train = sp.hstack((X_train, featuresTrain))
    
    
    (data_textTest, numURLtest) = pre_pross(data_textTest)
    X_test = vectorizer.transform(data_textTest)
    
    feature_names.extend(["punct", "upperC", "numEmo", "numChar", "numWord", "numExp", "numRepet", "numHashTags", "numRT","numURL"])
    
    featuresTest = add_features(data_textTest)
    featuresTest = np.column_stack((featuresTest, numURLtest))
    X_test = sp.hstack((X_test, featuresTest))
    
    return (X_train,X_test,feature_names)

def topFeatures(X_train, y_train,X_test,feature_names):
    
    ch2 = SelectKBest(chi2, k=20)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    
    return(feature_names)

def LR_class(data_textTrain, y_train, data_textTest, y_test, gram1, gram2):
    (X_train, X_test,feature_names) = vec_data(data_textTrain, data_textTest, gram1, gram2)
    parameters = {'penalty':['l1','l2'],'fit_intercept':('False', 'True'), 'C':[0.1,1,100],'tol':[1,.1, .01]}
    svc = LogisticRegression()
    clf = GridSearchCV(svc, parameters)
    clf.fit(X_train, y_train)
    print ("Accuracy Logistic Regression")
    print(clf.fit(X_train, y_train).score(X_test, y_test))
    
    features = topFeatures(X_train,y_train,X_test,feature_names)
    y_pred =clf.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    
    print("Contengency Table Logistic Regression")
    print(cnf_matrix)
    print("Top 20 Features Logistic Regression")
    print(features)

def svm_class(data_textTrain, y_train, data_textTest, y_test, gram1, gram2):
    (X_train, X_test, feature_names) = vec_data(data_textTrain, data_textTest, gram1, gram2)
    parameters = {'penalty':['l2'],'fit_intercept':('False', 'True'), 'C':[0.01,0.1,1,100],'tol':[1,.1, .01,.001]}
    svc = svm.LinearSVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(X_train, y_train)
    print ("Accuracy SVM")
    print(clf.fit(X_train, y_train).score(X_test, y_test))
    
    features = topFeatures(X_train,y_train,X_test,feature_names)
    y_pred =clf.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    
    print("Contengency Table SVM")
    print(cnf_matrix)
    print("Top 20 Features SVM")
    print(features)

def naiveBayes_class(data_textTrain, y_train, data_textTest, y_test, gram1, gram2):
    (X_train, X_test, feature_names) = vec_data(data_textTrain, data_textTest, gram1, gram2)
    parameters = {'fit_prior':['True','False'],'alpha':[1,0.1 ,0.001]}
    svc = MultinomialNB()
    clf = GridSearchCV(svc, parameters)
    clf.fit(X_train, y_train)
    print ("Accuracy Naive Bayes")
    print(clf.fit(X_train, y_train).score(X_test, y_test))
    
    features = topFeatures(X_train,y_train,X_test,feature_names)
    y_pred =clf.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    
    print("Contengency Table Naive Bayes")
    print(cnf_matrix)
    print("Top 20 Features Naive Bayes")
    print(features)


def best_model(trainFile, testFile):
    file_nameTrain = (trainFile)
    (data_textTrain, data_labelTrain)=read_file(file_nameTrain)
    y_train=[0 if x=='republican' else 1 for x in data_labelTrain]
    
    file_nameTest=(testFile)
    (data_textTest, data_labelTest)=read_file(file_nameTest)
    y_test=[0 if x=='republican' else 1 for x in data_labelTest]
    
    (X_train, X_test, feature_names) = vec_data(data_textTrain, data_textTest, 1, 1)
    
    clf = MultinomialNB(alpha=0.1, fit_prior=True)
    bestModel = clf.fit(X_train, y_train)
    print(bestModel.score(X_test, y_test))
    
    features = topFeatures(X_train, data_labelTrain,X_test,feature_names)
    
    # write a file
    file = 'vectorized_test_data.pkl'
    f = open(file, 'wb')
    pickle.dump((y_test,X_test,features), f)
    f.close()
    
    #save the object into a pickle
    filenameModel = 'model.pkl'
    pickle.dump(bestModel, open(filenameModel, 'wb'))


def main():
    file_nameTrain = sys.argv[1]
    file_nameTest = sys.argv[2]
    best_model(file_nameTrain,file_nameTest)

if __name__ == "__main__":
    main()
