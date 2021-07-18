from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def solve_SVC(x1,x2,x3,x4):
    Task = [[x1, x2, x3, x4]]   #Taken from the GUI input


    iris = load_iris()
    X = iris.data              #Features of the flowers
    Y = iris.target            #Species
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #Dividing the dataset to train and test


    model = svm.SVC(decision_function_shape='ovo')      #using multiclass classfying methods one vs one model

    model.fit(X_test, Y_test)                    #Training the model
    pred = model.predict(X_test)                 #Testing our model
    acc = accuracy_score(Y_test,pred)

    answer = model.predict(Task)            #predicting using trained model for our input

    if answer == 0:
        string = 'Iris-setosa'
    elif answer == 1:
        string = 'Iris-versicolor'
    else:
        string = 'Iris-virginica'
    return acc, string



















'''
import pandas as pd

df = pd.read_csv('iris.csv')
df = df.drop(['Id'],axis=1)
target = df['Species']
s = set()
for val in target:
    s.add(val)
s = list(s)
rows = list(range(100,150))
df = df.drop(df.index[rows])

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
## Drop rest of the features and extract the target values
Y = []
target = df['Species']
for val in target:
    if(val == 'Iris-setosa'):
        Y.append(0)
    elif(val == 'Iris-versicolor'):
        Y.append(1)
    else:
        Y.append(2)
df = df.drop(['Species'],axis=1)
X = df.values.tolist()
## Shuffle and split the data into training and test set
X, Y = shuffle(X,Y)
x_train = []
y_train = []
x_test = []
y_test = []

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

y_train = y_train.reshape(90,1)
y_test = y_test.reshape(10,1)

## Support Vector Machine
import numpy as np

train_f1 = x_train[:, 0]
train_f2 = x_train[:, 1]
train_f3 = x_train[:, 2]
train_f4 = x_train[:, 3]

train_f1 = train_f1.reshape(90, 1)
train_f2 = train_f2.reshape(90, 1)
train_f3 = train_f3.reshape(90, 1)
train_f4 = train_f4.reshape(90, 1)

w1 = np.zeros((90, 1))
w2 = np.zeros((90, 1))
w3 = np.zeros((90, 1))
w4 = np.zeros((90, 1))

epochs = 1
alpha = 0.0001

while (epochs < 10000):
    y = w1 * train_f1 + w2 * train_f2 + w3 * train_f3 + w4 * train_f4
    prod = y * y_train
    print(epochs)
    count = 0
    for val in prod:
        if (val >= 1):
            cost = 0
            w1 = w1 - alpha * (2 * 1 / epochs * w1)
            w2 = w2 - alpha * (2 * 1 / epochs * w2)
            w3 = w3 - alpha * (2 * 1 / epochs * w3)
            w4 = w4 - alpha * (2 * 1 / epochs * w4)

        else:
            cost = 1 - val
            w1 = w1 + alpha * (train_f1[count] * y_train[count] - 2 * 1 / epochs * w1)
            w2 = w2 + alpha * (train_f2[count] * y_train[count] - 2 * 1 / epochs * w2)
            w3 = w3 + alpha * (train_f3[count] * y_train[count] - 2 * 1 / epochs * w3)
            w4 = w4 + alpha * (train_f4[count] * y_train[count] - 2 * 1 / epochs * w4)
        count += 1
    epochs += 1

from sklearn.metrics import accuracy_score

## Clip the weights
index = list(range(10,90))
w1 = np.delete(w1,index)
w2 = np.delete(w2,index)
w3 = np.delete(w3,index)
w4 = np.delete(w4,index)

w1 = w1.reshape(10,1)
w2 = w2.reshape(10,1)
w3 = w3.reshape(10,1)
w4 = w4.reshape(10,1)
## Extract the test data features
test_f1 = x_test[:,0]
test_f2 = x_test[:,1]
test_f3 = x_test[:,2]
test_f4 = x_test[:,3]

test_f1 = test_f1.reshape(10,1)
test_f2 = test_f2.reshape(10,1)
test_f3 = test_f3.reshape(10,1)
test_f4 = test_f4.reshape(10,1)
## Predict
y_pred = w1 * test_f1 + w2 * test_f2 + w3 * test_f3 + w4 * test_f4
predictions = []
for val in y_pred:
    if(val == 0):
        predictions.append(0)
    elif(val==1):
        predictions.append(1)
    else:
        predictions.append(2)

print(accuracy_score(y_test,predictions))
'''

