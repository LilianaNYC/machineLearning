from classify import best_model, LR_class, read_file, naiveBayes_class, svm_class
from analyze import mainFun


file_nameTrain = ('/Users/lilianacruzlopez/Desktop/NLP/HW/HW1/train.txt')
file_nameTest = ('/Users/lilianacruzlopez/Desktop/NLP/HW/HW1/dev.txt')

(data_textTrain, data_labelTrain) = read_file(file_nameTrain)
y_train=[0 if x=='republican' else 1 for x in data_labelTrain]

(data_textTest, data_labelTest)=read_file(file_nameTest)
y_test=[0 if x=='republican' else 1 for x in data_labelTest]

#LR_class(data_textTrain, y_train, data_textTest, y_test, 1,1)
print("Unigram")
naiveBayes_class(data_textTrain, y_train, data_textTest, y_test, 1,1)
print("Bigram")
naiveBayes_class(data_textTrain, y_train, data_textTest, y_test, 2,2)
print("Trigram")
naiveBayes_class(data_textTrain, y_train, data_textTest, y_test, 3,3)
#svm_class(data_textTrain, y_train, data_textTest, y_test, 1,1)


#calls classify for the best model
print("Best Model Accuracy")
best_model(file_nameTrain,file_nameTest)

print("Best Model Info")
#calls analyze
mainFun()
