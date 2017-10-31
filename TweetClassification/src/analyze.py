import pickle
from sklearn.metrics import confusion_matrix

def mainFun():
    # load the model from disk
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    (y_test,X_test, feature_names) = pickle.load(open('vectorized_test_data.pkl', 'rb'))
    y_pred =loaded_model.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    
    print("Contengency Table")
    print(cnf_matrix)
    print("Top 20 Features")
    print(feature_names)







