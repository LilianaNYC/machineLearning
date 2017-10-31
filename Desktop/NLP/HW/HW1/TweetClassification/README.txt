Name: Liliana Cruz-Lopez
Email: lc3211@columbia.edu

Description: In this project I built a classifier that will predict whether a person is a Democrat or a Republican depending on their Twitter posts.

Instructions (classify.py):
In order to make classify.py work successfully you need run it from the command using the following syntax:
	python3 classify.py trainData.txt testData.txt
the output should be a number that represents the accuracy of the best model. In addition this file will create two pickle files. One of them is named model.pkl and has the trained model. The other pickle file is vectorized_test_data.pkl that contains the all the test data information. 
Libraries used (classify.py):
csv
sklearn
numpy 
re
string
scipy
pickle

Instructions (tweets.py):
In order to make tweets.py work successfully you need to run it from command line using the following syntax:
	python3 tweets.py
WARNING: change the path files here for the train and test data (It will not compile successfully if this is not modified). The output should be the accuracy, confusion matrix, and top 20 features for a unigram, bigram, and trigram. In addition, it calls classify.py and analyze.py to display the accuracy, confusion matrix, and top 20 features of the best model. Also, the unigram, bigram, and trigram runs Naive Bayes Classifier. If you are interested in learning about the performance of the other classifiers please replace these functions with the following.
LR_class(data_textTrain, y_train, data_textTest, y_test, 1,1)  ----> to run Logistic Regression model on unigram (the grams can be adjusted as you please)
naiveBayes_class(data_textTrain, y_train, data_textTest, y_test, 2,2) ----> to run Multinomial Naives Bayes on a bigram (the grams can be adjusted as you please) 
svm_class(data_textTrain, y_train, data_textTest, y_test, 3,3) -----> to run LinearSVC SVM on trigram  (the grams can be adjusted as you please)
Libraries used (tweets.py):
It call classify and analyze python files

Instructions (analyze.py): 
In order to make tweets.py work successfully you need to train the model from classify.py first and the call this file from tweets.py to display the confusion matrix and top 20 features of the best model. This file will contain the two pickle files created by classify.py. 
Libraries used (analyze.py):
pickle
sklearn

All the libraries used in this assignment were installed before or came with the latest version of python 3. I did not installed anything. 

This project was written using functions to make things easier, reduce redundancy, and decrease minor errors. You can modify the number of features by modifying the add_features function. Also, you can modify the pre-processing of the data by modifying pre_pross function.  
