import sklearn.discriminant_analysis
from  sklearn import  datasets
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier


def solve_lda(x1,x2,x3,x4):
    query = [[x1,x2,x3,x4]]
    iris=datasets.load_iris()
    #print(iris.data)
    #print(iris.target)
    x=iris.data
    y=iris.target
    #x has features and y has labels
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)
    # print(x_test)
    #building


    classifier=sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    # classifier = OneVsRestClassifier(tree.DecisionTreeClassifier())
    classifier.fit(x_train,y_train)
    predictions=classifier.predict(x_test)


    acc=(accuracy_score(y_test,predictions))
    prediction=(classifier.predict(query))
    pred = classifier.predict_proba(query)
    if(prediction==0):
        string="Iris-setosa"
    elif(prediction==1):
        string='Iris-versicolor'
    else:
        string='Iris-virginica'
    return(acc,string)
